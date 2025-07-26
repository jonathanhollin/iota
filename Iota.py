#!/usr/bin/env python3
"""
Composer Symphony Iota Calculator

Dependencies:
    pip install pandas numpy yfinance scipy
    Optional: pip install tqdm matplotlib
    A helper module `sim.py` with `fetch_backtest()` & `calculate_portfolio_returns()` must be importable.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import warnings
import threading
import time

import numpy as np
import pandas as pd
from scipy import stats

# Optional imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sim import fetch_backtest, calculate_portfolio_returns
except ImportError:
    sys.exit("ERROR: Cannot import helpers from sim.py. Ensure sim.py is on PYTHONPATH.")

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

def parse_exclusion_input(user_str: str) -> List[Tuple[datetime.date, datetime.date]]:
    """Return list of date ranges from user string."""
    if not user_str.strip():
        return []
    out: List[Tuple[datetime.date, datetime.date]] = []
    for token in user_str.split(","):
        token = token.strip()
        if not token:
            continue
        found = re.findall(r"\d{4}-\d{2}-\d{2}", token)
        if len(found) != 2:
            print(f"[!] Skipping unparsable exclusion token: '{token}'.")
            continue
        a, b = [datetime.strptime(d, "%Y-%m-%d").date() for d in found]
        out.append((min(a, b), max(a, b)))
    return out

def spinner_animation(message: str, stop_event: threading.Event):
    """Display a spinning animation while fetching data."""
    if not HAS_TQDM:
        return
    
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop_event.is_set():
        print(f"\r{message} {spinner_chars[i % len(spinner_chars)]}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r{message} ✓", flush=True)

def fetch_with_progress(url: str, early: str, today: str, verbose: bool = True):
    """Fetch backtest data with progress indication."""
    if not verbose:
        return fetch_backtest(url, early, today)
    
    if HAS_TQDM:
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=spinner_animation, 
                                         args=("Fetching back‑test data", stop_spinner))
        spinner_thread.daemon = True
        spinner_thread.start()
        
        try:
            result = fetch_backtest(url, early, today)
            stop_spinner.set()
            spinner_thread.join(timeout=0.2)
            return result
        except Exception as e:
            stop_spinner.set()
            spinner_thread.join(timeout=0.2)
            print("\rFetching back‑test data ✗")
            raise e
    else:
        print("Fetching back‑test data …")
        return fetch_backtest(url, early, today)

def summarize(series: pd.Series) -> Tuple[float, float, float]:
    """Return (q25, median, q75) rounded to native float."""
    return tuple(float(series.quantile(q)) for q in (0.25, 0.5, 0.75))

def cumulative_return(daily_pct: pd.Series) -> float:
    """Total compounded return over the period (decimal)."""
    daily_dec = daily_pct.dropna() / 100.0
    return float(np.prod(1 + daily_dec) - 1) if not daily_dec.empty else 0.0

def window_cagr(daily_pct: pd.Series) -> float:
    """Compounded annual growth rate over window - FIXED to return decimal (not percent)."""
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.empty:
        return 0.0
    total_return = np.prod(1 + daily_dec) - 1
    days = len(daily_dec)
    if days < 2:
        return 0.0
    try:
        cagr = (1 + total_return) ** (252 / days) - 1
        return cagr  # Return as decimal, not percent
    except (FloatingPointError, ValueError):
        return 0.0
    
def sharpe_ratio(daily_pct: pd.Series) -> float:
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.std(ddof=0) == 0:
        return 0.0
    return (daily_dec.mean() / daily_dec.std(ddof=0)) * np.sqrt(252)

def sortino_ratio(daily_pct: pd.Series) -> float:
    """Enhanced Sortino ratio with proper zero-downside handling."""
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.empty:
        return 0.0
    
    downside = daily_dec[daily_dec < 0]
    mean_return = daily_dec.mean()
    
    if len(downside) == 0:
        if mean_return > 0:
            return np.inf  # Perfect risk-adjusted return
        else:
            return 0.0  # No return, no risk
    
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return 0.0
    
    return (mean_return / downside_std) * np.sqrt(252)

def assess_sample_reliability(n_is: int, n_oos: int) -> str:
    """Assess statistical reliability based on sample sizes using statistical principles."""
    min_size = min(n_is, n_oos)
    
    if min_size >= 378:  # ~1.5 years - high statistical power
        return "HIGH_CONFIDENCE"
    elif min_size >= 189:  # ~9 months - moderate statistical power
        return "MODERATE_CONFIDENCE"  
    elif min_size >= 90:   # ~4.5 months - low but usable statistical power
        return "LOW_CONFIDENCE"
    else:                  # <90 days - insufficient for reliable statistics
        return "INSUFFICIENT_DATA"

def format_sortino_output(sortino_val: float) -> str:
    """Special formatting for Sortino ratio including infinite values."""
    if np.isinf(sortino_val):
        return "∞ (no downside)"
    elif np.isnan(sortino_val):
        return "NaN"
    else:
        return f"{sortino_val:.3f}"

def build_slices(is_ret: pd.Series, slice_len: int, n_slices: int, overlap: bool) -> List[pd.Series]:
    """Return list of IS slices each of length slice_len."""
    total_is = len(is_ret)
    max_start = total_is - slice_len

    if max_start < 0:
        return []

    if not overlap:
        slices: List[pd.Series] = []
        end_idx = total_is
        while len(slices) < n_slices and end_idx >= slice_len:
            seg = is_ret.iloc[end_idx - slice_len : end_idx]
            if len(seg) == slice_len:
                slices.append(seg)
            end_idx -= slice_len
        return slices

    if n_slices == 1:
        starts = [max_start]
    else:
        starts = np.linspace(0, max_start, n_slices, dtype=int).tolist()
    starts = sorted(dict.fromkeys(starts))

    return [is_ret.iloc[s : s + slice_len] for s in starts]

def compute_iota(is_metric: float, oos_metric: float, n_oos: int, n_ref: int = 252, eps: float = 1e-6, 
                 lower_is_better: bool = False, is_values: np.ndarray = None) -> float:
    """
    INTUITIVE standardized iota calculation where:
    - Positive iota = Better OOS performance  
    - Negative iota = Worse OOS performance
    
    Uses standardized approach: ι = w * (OOS - IS_median) / IS_std_dev
    This makes iota directly interpretable as "weighted standard deviations from IS median"
    """
    # Handle infinite values (e.g., from Sortino with no downside)
    if np.isinf(oos_metric):
        return 2.0 if not lower_is_better else -2.0  # OOS is "perfect"
    
    if is_values is None:
        # Fallback to old method if IS values not provided
        return compute_iota_legacy(is_metric, oos_metric, n_oos, n_ref, eps, lower_is_better)
    
    # Remove infinite values from IS data
    finite_is = is_values[np.isfinite(is_values)]
    if len(finite_is) < 2:
        return 0.0  # Insufficient data for standardization
    
    is_median = np.median(finite_is)
    is_std = np.std(finite_is, ddof=1)  # Sample standard deviation
    
    if is_std < eps:
        return 0.0  # No variation in IS data
    
    # Calculate standardized difference
    standardized_diff = (oos_metric - is_median) / is_std
    
    # For "lower is better" metrics, flip the sign
    if lower_is_better:
        standardized_diff = -standardized_diff
    
    # Apply sample size weighting
    w = min(1.0, np.sqrt(n_oos / n_ref))
    
    return w * standardized_diff

def compute_iota_legacy(is_metric: float, oos_metric: float, n_oos: int, n_ref: int = 252, eps: float = 1e-6, 
                       lower_is_better: bool = False) -> float:
    """Legacy log-ratio iota calculation for backward compatibility."""
    # Handle infinite values
    if np.isinf(is_metric) or np.isinf(oos_metric):
        if np.isinf(is_metric) and np.isinf(oos_metric):
            return 0.0  # Both infinite - no difference
        elif np.isinf(oos_metric):
            return 2.0 if not lower_is_better else -2.0  # OOS is "perfect"
        else:
            return -2.0 if not lower_is_better else 2.0  # IS was "perfect"
    
    x = max(abs(is_metric), eps)
    y = max(abs(oos_metric), eps)
    
    # Handle negative values properly
    if is_metric < 0 and oos_metric < 0:
        # Both negative: smaller absolute value is "better"
        x, y = abs(is_metric), abs(oos_metric)
    elif is_metric < 0 or oos_metric < 0:
        # One negative: significant regime change
        return -2.0 if oos_metric < is_metric else 2.0
    
    # For "lower is better" metrics, flip the ratio
    if lower_is_better:
        x, y = y, x
    
    w = min(1.0, np.sqrt(n_oos / n_ref))
    
    # Use OOS/IS ratio to make positive = better performance
    return w * np.log(y / x)

def iota_to_persistence_rating(iota_val: float, max_rating: int = 500) -> int:
    """Convert INTUITIVE standardized iota to persistence rating with proper scaling."""
    if not np.isfinite(iota_val):
        return 100
    
    # Adjusted exponential transformation for standardized iota
    # Using k = 0.5 so that ±1σ gives more reasonable rating spread
    k = 0.5
    rating = 100 * np.exp(k * iota_val)
    return max(1, min(max_rating, int(round(rating))))

def interpret_iota_directly(iota_val: float) -> str:
    """Direct interpretation of standardized iota values."""
    if not np.isfinite(iota_val):
        return "UNDEFINED"
    
    if iota_val >= 2.0:
        return "EXCEPTIONAL: OOS >2σ above IS median (ι ≥ +2.0)"
    elif iota_val >= 1.0:
        return "EXCELLENT: OOS >1σ above IS median (ι ≥ +1.0)"
    elif iota_val >= 0.5:
        return "GOOD: OOS >0.5σ above IS median (ι ≥ +0.5)"  
    elif iota_val >= 0.1:
        return "SLIGHT_IMPROVEMENT: OOS mildly above IS median (ι ≥ +0.1)"
    elif iota_val >= -0.1:
        return "NEUTRAL: OOS ≈ IS median (-0.1 ≤ ι ≤ +0.1)"
    elif iota_val >= -0.5:
        return "CAUTION: OOS below IS median (ι ≤ -0.1)"
    elif iota_val >= -1.0:
        return "WARNING: OOS >0.5σ below IS median (ι ≤ -0.5)"
    elif iota_val >= -2.0:
        return "ALERT: OOS >1σ below IS median (ι ≤ -1.0)"
    else:
        return "CRITICAL: OOS >2σ below IS median (ι ≤ -2.0)"

def interpret_persistence_rating(rating: int, confidence_interval: Tuple[float, float] = None) -> str:
    """Generate interpretation text for persistence rating aligned with standardized iota."""
    if rating >= 165:  # ι ≈ +1.0
        interpretation = "EXCELLENT: Significant OOS outperformance (>1σ)"
    elif rating >= 135:  # ι ≈ +0.6
        interpretation = "GOOD: Notable OOS outperformance (>0.5σ)"
    elif rating >= 105:  # ι ≈ +0.1
        interpretation = "SLIGHT_OUTPERFORMANCE: Mild OOS improvement"
    elif rating >= 95:   # ι ≈ -0.1
        interpretation = "NEUTRAL: OOS ≈ IS median (-0.1 ≤ ι ≤ +0.1)"
    elif rating >= 85:   # ι ≈ -0.3
        interpretation = "CAUTION: OOS below IS median"
    elif rating >= 75:   # ι ≈ -0.5
        interpretation = "WARNING: OOS >0.5σ below IS median"
    elif rating >= 60:   # ι ≈ -1.0
        interpretation = "ALERT: OOS >1σ below IS median"
    else:                # ι < -1.0
        interpretation = "CRITICAL: OOS >1σ below IS median"
    
    # Add confidence qualifier if available
    if confidence_interval and all(np.isfinite(confidence_interval)):
        ci_lower, ci_upper = confidence_interval
        ci_lower_rating = iota_to_persistence_rating(ci_lower)
        ci_upper_rating = iota_to_persistence_rating(ci_upper)
        
        if ci_lower_rating < 95 and ci_upper_rating > 105:
            interpretation += " (HIGH_UNCERTAINTY)"
        elif abs(ci_upper_rating - ci_lower_rating) > 50:
            interpretation += " (MODERATE_UNCERTAINTY)"
    
    return interpretation

# ===== ENHANCED BOOTSTRAP METHODS WITH AUTOCORRELATION ADJUSTMENT =====
def standard_bootstrap_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                                 n_bootstrap: int, confidence_level: float, 
                                 lower_is_better: bool, verbose: bool) -> Tuple[float, float]:
    """Standard bootstrap for non-overlapping slices."""
    try:
        bootstrap_iotas = []
        for _ in range(n_bootstrap):
            try:
                boot_sample = np.random.choice(is_values, size=len(is_values), replace=True)
                boot_median = np.median(boot_sample)
                boot_iota = compute_iota(boot_median, oos_value, n_oos, lower_is_better=lower_is_better, 
                                       is_values=boot_sample)
                if np.isfinite(boot_iota):
                    bootstrap_iotas.append(boot_iota)
            except Exception:
                continue
        
        if len(bootstrap_iotas) < 50:
            return np.nan, np.nan
        
        alpha = 1 - confidence_level
        return tuple(np.percentile(bootstrap_iotas, [100 * alpha/2, 100 * (1 - alpha/2)]))
    except Exception:
        return np.nan, np.nan

def block_bootstrap_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                              n_bootstrap: int, confidence_level: float, 
                              lower_is_better: bool, verbose: bool) -> Tuple[float, float]:
    """Block bootstrap for overlapping slices."""
    try:
        bootstrap_iotas = []
        block_size = max(3, min(len(is_values) // 8, 20))
        
        for _ in range(n_bootstrap):
            try:
                boot_sample = []
                n_blocks_needed = (len(is_values) // block_size) + 1
                for _ in range(n_blocks_needed):
                    if len(is_values) <= block_size:
                        boot_sample.extend(is_values)
                    else:
                        start_idx = np.random.randint(0, len(is_values) - block_size + 1)
                        block = is_values[start_idx:start_idx + block_size]
                        boot_sample.extend(block)
                
                boot_sample = np.array(boot_sample[:len(is_values)])
                if len(boot_sample) > 0:
                    boot_median = np.median(boot_sample)
                    boot_iota = compute_iota(boot_median, oos_value, n_oos, 
                                           lower_is_better=lower_is_better, is_values=boot_sample)
                    if np.isfinite(boot_iota):
                        bootstrap_iotas.append(boot_iota)
            except Exception:
                continue
        
        if len(bootstrap_iotas) < 50:
            return np.nan, np.nan
        
        alpha = 1 - confidence_level
        return tuple(np.percentile(bootstrap_iotas, [100 * alpha/2, 100 * (1 - alpha/2)]))
    except Exception:
        return np.nan, np.nan

def bootstrap_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int, 
                             n_bootstrap: int = 1000, confidence_level: float = 0.95,
                             lower_is_better: bool = False, verbose: bool = True,
                             overlap: bool = True) -> Tuple[float, float]:
    """Enhanced bootstrap confidence interval with block bootstrap for overlapping slices."""
    if len(is_values) < 3:
        return np.nan, np.nan
    
    # For very small samples, use parametric approach
    if len(is_values) < 10:
        return parametric_iota_confidence(is_values, oos_value, n_oos, 
                                        confidence_level, lower_is_better)
    
    # Choose bootstrap method based on overlap
    if overlap:
        return block_bootstrap_confidence(is_values, oos_value, n_oos, n_bootstrap, 
                                        confidence_level, lower_is_better, verbose)
    else:
        return standard_bootstrap_confidence(is_values, oos_value, n_oos, n_bootstrap, 
                                           confidence_level, lower_is_better, verbose)

def parametric_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                              confidence_level: float = 0.95, 
                              lower_is_better: bool = False) -> Tuple[float, float]:
    """Parametric confidence interval using t-distribution."""
    if len(is_values) < 3:
        return np.nan, np.nan
    
    # Compute iota for each IS value vs OOS
    iotas = [compute_iota(is_val, oos_value, n_oos, lower_is_better=lower_is_better, 
                         is_values=np.array([is_val])) for is_val in is_values]
    iotas = [i for i in iotas if np.isfinite(i)]
    
    if len(iotas) < 3:
        return np.nan, np.nan
    
    mean_iota = np.mean(iotas)
    se_iota = stats.sem(iotas)
    dof = len(iotas) - 1
    
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, dof)
    margin = t_critical * se_iota
    
    return mean_iota - margin, mean_iota + margin

def calculate_autocorrelation_adjustment(values: np.ndarray, overlap: bool) -> float:
    """
    Calculate autocorrelation adjustment factor for overlapping slice p-values.
    
    For overlapping slices, we need to adjust the effective sample size due to 
    temporal correlation. This uses the approach from Brandt & Diebold (2006)
    for overlapping observations.
    """
    if not overlap or len(values) < 3:
        return 1.0  # No adjustment needed for non-overlapping data
    
    try:
        # Calculate first-order autocorrelation of the input values
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        
        # Handle NaN or extreme values
        if not np.isfinite(autocorr):
            autocorr = 0.0
        autocorr = np.clip(autocorr, -0.99, 0.99)  # Prevent division by zero
        
        # Effective sample size adjustment for overlapping data
        # Based on Newey-West type correction for overlapping observations
        n = len(values)
        if autocorr > 0:
            # Positive autocorrelation reduces effective sample size
            effective_n = n * (1 - autocorr) / (1 + autocorr)
            adjustment_factor = np.sqrt(effective_n / n)
        else:
            # Negative autocorrelation increases effective sample size (conservative: no adjustment)
            adjustment_factor = 1.0
        
        # Ensure adjustment factor is reasonable (between 0.1 and 1.0)
        adjustment_factor = np.clip(adjustment_factor, 0.1, 1.0)
        
        return adjustment_factor
        
    except Exception:
        # If anything goes wrong, return conservative adjustment
        return 0.7  # Conservative adjustment assuming moderate positive autocorrelation

def wilcoxon_iota_test_autocorr_adjusted(is_values: np.ndarray, oos_value: float, n_oos: int,
                                        lower_is_better: bool = False, overlap: bool = True) -> Tuple[float, bool]:
    """
    Wilcoxon test on slice-level iotas with autocorrelation adjustment for overlapping slices.
    """
    if len(is_values) < 6:  # Minimum for meaningful Wilcoxon test
        return np.nan, False
    
    # Compute iota for each IS slice
    slice_iotas = []
    for is_val in is_values:
        iota_val = compute_iota(is_val, oos_value, n_oos, lower_is_better=lower_is_better, 
                               is_values=is_values)
        if np.isfinite(iota_val):
            slice_iotas.append(iota_val)
    
    if len(slice_iotas) < 6:
        return np.nan, False
    
    try:
        # Standard Wilcoxon test
        _, p_value_raw = stats.wilcoxon(slice_iotas, alternative='two-sided')
        
        # Apply autocorrelation adjustment for overlapping slices
        if overlap:
            # Calculate autocorrelation on the SLICE IOTAS, not raw IS values
            autocorr_adjustment = calculate_autocorrelation_adjustment(np.array(slice_iotas), overlap)
            
            # CORRECTED: Multiply p-value to make it more conservative
            # Lower adjustment factors should make p-values LARGER (less significant)
            p_value_adjusted = min(1.0, p_value_raw / autocorr_adjustment)
            
            return p_value_adjusted, p_value_adjusted < 0.05
        else:
            # No adjustment needed for non-overlapping slices
            return p_value_raw, p_value_raw < 0.05
            
    except (ValueError, ZeroDivisionError):
        return np.nan, False
    
def compute_iota_with_stats(is_values: np.ndarray, oos_value: float, n_oos: int, 
                           metric_name: str = "metric", lower_is_better: bool = False,
                           verbose: bool = True, overlap: bool = True) -> Dict[str, Any]:
    """Enhanced iota computation with autocorrelation-adjusted statistical tests."""
    if len(is_values) == 0:
        return {
            'iota': np.nan,
            'persistence_rating': 100,
            'confidence_interval': (np.nan, np.nan),
            'p_value_adjusted': np.nan,
            'significant': False,
            'median_is': np.nan,
            'iqr_is': (np.nan, np.nan),
            'test_method': 'none',
            'bootstrap_method': 'none',
            'autocorr_adjustment': 1.0
        }
    
    # Basic statistics
    median_is = np.median(is_values)
    q25_is, q75_is = np.percentile(is_values, [25, 75])
    
    # Compute iota with direction awareness
    iota = compute_iota(median_is, oos_value, n_oos, lower_is_better=lower_is_better, is_values=is_values)
    persistence_rating = iota_to_persistence_rating(iota)
    
    # Enhanced bootstrap confidence interval with overlap awareness
    ci_lower, ci_upper = bootstrap_iota_confidence(is_values, oos_value, n_oos, 
                                                  lower_is_better=lower_is_better, 
                                                  verbose=verbose, overlap=overlap)
    
    # Autocorrelation-adjusted statistical test
    p_value_adjusted, significant = wilcoxon_iota_test_autocorr_adjusted(is_values, oos_value, n_oos, 
                                                                        lower_is_better=lower_is_better,
                                                                        overlap=overlap)
    
    # CORRECTED: Calculate autocorrelation adjustment factor on IS values for reporting
    # (This is just for display purposes - the actual adjustment is done in the test function)
    if overlap:
        # For reporting, calculate autocorrelation on IS values
        autocorr_adjustment = calculate_autocorrelation_adjustment(is_values, overlap)
    else:
        autocorr_adjustment = 1.0
    
    test_method = 'wilcoxon_autocorr_adjusted' if overlap else 'wilcoxon_standard'
    bootstrap_method = 'block_bootstrap' if overlap else 'standard_bootstrap'
    
    return {
        'iota': iota,
        'persistence_rating': persistence_rating,
        'confidence_interval': (ci_lower, ci_upper),
        'p_value_adjusted': p_value_adjusted,
        'significant': significant,
        'median_is': median_is,
        'iqr_is': (q25_is, q75_is),
        'test_method': test_method,
        'bootstrap_method': bootstrap_method,
        'autocorr_adjustment': autocorr_adjustment
    }
# ===== END ENHANCED BOOTSTRAP METHODS =====

def rolling_oos_analysis(daily_ret: pd.Series, oos_start_dt: datetime.date, 
                        is_ret: pd.Series, n_slices: int = 100, overlap: bool = True,
                        window_size: int = None, step_size: int = None, 
                        min_windows: int = 6, verbose: bool = True) -> Dict[str, Any]:
    """
    Simplified rolling analysis without composite iota and drawdown components.
    """
    # Get data from OOS start onwards
    oos_data = daily_ret[daily_ret.index >= oos_start_dt]
    total_oos_days = len(oos_data)
    
    # SAFETY: Prevent crashes with very long periods
    if total_oos_days > 1500:  # ~6 years
        if verbose:
            print(f"  ⚠️  Very long OOS period ({total_oos_days} days) - using last 1000 days to prevent crashes")
        oos_data = oos_data.iloc[-1000:]
        total_oos_days = len(oos_data)
    
    if total_oos_days < 90:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # ADAPTIVE WINDOW SIZING based on OOS period length
    if window_size is None:
        if total_oos_days >= 504:  # 2+ years
            window_size = 126  # 6 months
        elif total_oos_days >= 252:  # 1-2 years  
            window_size = 84   # 4 months
        elif total_oos_days >= 189:  # 9+ months
            window_size = 63   # 3 months
        else:  # 3-9 months
            window_size = max(21, total_oos_days // 4)
    
    # ADAPTIVE STEP SIZE for smoother analysis
    if step_size is None:
        step_size = max(5, window_size // 8)
    
    if total_oos_days < window_size + step_size:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Calculate expected number of windows with safety limit
    max_possible_windows = (total_oos_days - window_size) // step_size + 1
    max_windows = min(60, max_possible_windows)  # Cap at 60 windows to prevent crashes
    
    if verbose and max_windows >= 10:
        print(f"  Creating {max_windows} rolling windows ({window_size} days each, {step_size} day steps)")
    
    # Create IS slices that match rolling window size
    is_slices = build_slices(is_ret, window_size, n_slices, overlap)
    if not is_slices:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_IS_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Pre-compute IS metrics once (performance optimization) - REMOVED MAX DRAWDOWN
    if verbose and max_windows >= 15:
        print("  Pre-computing IS slice metrics …")
    
    # FIXED: Removed 'ar' from metrics to avoid annualized return inflation IN ROLLING ANALYSIS ONLY
    is_metrics = {
        'sh': [sharpe_ratio(s) for s in is_slices], 
        'cr': [cumulative_return(s) for s in is_slices],
        'so': [sortino_ratio(s) for s in is_slices]
    }
    
    # Create rolling windows
    windows = []
    start_idx = 0
    window_count = 0
    
    if verbose and max_windows >= 15:
        print("  Computing rolling window iotas …")
    
    while start_idx + window_size <= len(oos_data) and window_count < max_windows:
        window_data = oos_data.iloc[start_idx:start_idx + window_size]
        if len(window_data) == window_size:
            window_num = len(windows) + 1
            
            # Calculate metrics for this OOS window - REMOVED ANNUALIZED RETURN FOR ROLLING ANALYSIS
            window_sh = sharpe_ratio(window_data)
            window_cr = cumulative_return(window_data)
            window_so = sortino_ratio(window_data)
            
            # Calculate iota for each metric using pre-computed IS metrics - REMOVED 'ar' FOR ROLLING ANALYSIS
            window_iotas = {}
            for metric in ['sh', 'cr', 'so']:  # REMOVED 'ar' from rolling analysis only
                is_values = np.array(is_metrics[metric])
                oos_value = {'sh': window_sh, 'cr': window_cr, 'so': window_so}[metric]
                lower_is_better = False  # All remaining metrics are "higher is better"
                
                if len(is_values) > 0 and np.isfinite(oos_value):
                    iota = compute_iota(np.median(is_values), oos_value, window_size, 
                                      lower_is_better=lower_is_better, is_values=is_values)
                    window_iotas[metric] = iota
                else:
                    window_iotas[metric] = np.nan
            
            windows.append({
                'start_date': window_data.index[0],
                'end_date': window_data.index[-1],
                'window_num': window_num,
                'returns': window_data,
                'metrics': {
                    'sh': window_sh, 
                    'cr': window_cr,
                    'so': window_so
                },
                'iotas': window_iotas
            })
            window_count += 1
        start_idx += step_size
    
    if len(windows) < min_windows:
        return {
            'sufficient_data': False,
            'n_windows': len(windows),
            'overfitting_risk': 'INSUFFICIENT_WINDOWS',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Extract iota series for analysis - REMOVED COMPOSITE IOTA
    window_nums = np.array([w['window_num'] for w in windows])
    
    # Get individual metric iotas - REMOVED 'ar' FROM ROLLING ANALYSIS ONLY
    metric_iotas = {}
    for metric in ['sh', 'cr', 'so']:
        metric_iotas[metric] = np.array([w['iotas'][metric] for w in windows if np.isfinite(w['iotas'][metric])])
    
    # Calculate trend slopes for individual metrics - REMOVED 'ar' FROM ROLLING ANALYSIS ONLY
    metric_slopes = {}
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) >= 3:
            try:
                slope, _, _, _, _ = stats.linregress(window_nums[:len(metric_iotas[metric])], metric_iotas[metric])
                metric_slopes[f'{metric}_slope'] = slope
            except:
                metric_slopes[f'{metric}_slope'] = np.nan
        else:
            metric_slopes[f'{metric}_slope'] = np.nan

    # Calculate average trend slope across all metrics as overall indicator
    valid_slopes = [slope for slope in metric_slopes.values() if np.isfinite(slope)]
    avg_trend_slope = np.mean(valid_slopes) if valid_slopes else np.nan
    
    # Calculate degradation score based on individual metric trends AND absolute performance
    degradation_score = 0

    # Collect all valid iotas across metrics for absolute performance assessment - REMOVED 'ar' FROM ROLLING ANALYSIS ONLY
    all_iotas = []
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) > 0:
            all_iotas.extend(metric_iotas[metric])

    if len(all_iotas) > 0:
        all_iotas = np.array(all_iotas)
        
        # ABSOLUTE PERFORMANCE PENALTIES (NEW)
        avg_iota = np.mean(all_iotas)
        if avg_iota < -1.5:
            degradation_score += 4  # Severely poor performance
        elif avg_iota < -1.0:
            degradation_score += 3  # Consistently poor performance
        elif avg_iota < -0.5:
            degradation_score += 2  # Moderately poor performance
        elif avg_iota < -0.2:
            degradation_score += 1  # Mildly poor performance
        
        # PROPORTION OF TIME BELOW EXPECTATIONS (NEW)
        negative_proportion = np.mean(all_iotas < 0)
        if negative_proportion > 0.9:
            degradation_score += 3  # Almost always underperforming
        elif negative_proportion > 0.75:
            degradation_score += 2  # Usually underperforming
        elif negative_proportion > 0.6:
            degradation_score += 1  # Often underperforming
        
        # SEVERITY OF UNDERPERFORMANCE (NEW)
        severely_negative = np.mean(all_iotas < -1.0)
        if severely_negative > 0.5:  # 50% of time severely underperforming
            degradation_score += 3
        elif severely_negative > 0.3:  # 30% of time severely underperforming
            degradation_score += 2
        elif severely_negative > 0.1:  # 10% of time severely underperforming
            degradation_score += 1

    # Check individual metric trends (EXISTING LOGIC - KEPT) - REMOVED 'ar' FROM ROLLING ANALYSIS ONLY
    for metric in ['sh', 'cr', 'so']:
        slope = metric_slopes.get(f'{metric}_slope', np.nan)
        if np.isfinite(slope):
            if slope < -0.15:
                degradation_score += 3
            elif slope < -0.08:
                degradation_score += 2
            elif slope < -0.03:
                degradation_score += 1

    # Check volatility across metrics (EXISTING LOGIC - KEPT) - REMOVED 'ar' FROM ROLLING ANALYSIS ONLY
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) > 2:
            iota_volatility = np.std(metric_iotas[metric])
            if iota_volatility > 0.8:
                degradation_score += 1

    # Check for performance deterioration over time (EXISTING LOGIC - KEPT) - REMOVED 'ar' FROM ROLLING ANALYSIS ONLY
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) >= 4:
            first_half = metric_iotas[metric][:len(metric_iotas[metric])//2]
            second_half = metric_iotas[metric][len(metric_iotas[metric])//2:]
            if len(first_half) > 0 and len(second_half) > 0:
                if np.mean(second_half) < np.mean(first_half) - 0.2:
                    degradation_score += 1
    
    # Assess overfitting risk (UPDATED THRESHOLDS)
    if degradation_score >= 12:
        risk_level = "CRITICAL"
    elif degradation_score >= 8:
        risk_level = "HIGH"
    elif degradation_score >= 5:
        risk_level = "MODERATE"
    elif degradation_score >= 2:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"    
    return {
        'sufficient_data': True,
        'n_windows': len(windows),
        'windows': windows,
        'iota_trend_slope': avg_trend_slope,
        'metric_slopes': metric_slopes,
        'degradation_score': degradation_score,
        'overfitting_risk': risk_level,
        'window_size_days': window_size,
        'step_size_days': step_size,
        'total_oos_days': total_oos_days,
        'is_slices_used': len(is_slices)
        }

def interpret_overfitting_risk(rolling_results: Dict[str, Any]) -> str:
    """Generate interpretation of backtest-based overfitting risk analysis."""
    if not rolling_results.get('sufficient_data', False):
        return "Insufficient data for overfitting analysis (need longer OOS period)"
    
    risk = rolling_results['overfitting_risk']
    n_windows = rolling_results['n_windows']
    avg_trend_slope = rolling_results.get('iota_trend_slope', np.nan)
    
    interpretation = f"{risk} overfitting risk based on {n_windows} rolling windows. "
    
    if risk == "CRITICAL":
        interpretation += "🚨 CRITICAL: Strategy severely degrading relative to backtest."
    elif risk == "HIGH":
        interpretation += "⚠️  HIGH: Strategy performance increasingly deviates from backtest expectations"
    elif risk == "MODERATE":  
        interpretation += "⚠️  MODERATE: Some degradation in backtest-to-OOS match detected - monitor closely."
    elif risk == "LOW":
        interpretation += "Minor inconsistencies with backtest - generally acceptable variation."
    else:
        interpretation += "✓ Consistent performance relative to backtest expectations - low overfitting concern."
    
    # Add specific trend information
    if np.isfinite(avg_trend_slope):
        if avg_trend_slope < -0.15:
            interpretation += f" Average iota declining rapidly at {avg_trend_slope:.3f} per window (severe degradation)."
        elif avg_trend_slope < -0.08:
            interpretation += f" Average iota declining at {avg_trend_slope:.3f} per window (moderate degradation)."
        elif avg_trend_slope < -0.03:
            interpretation += f" Average iota declining at {avg_trend_slope:.3f} per window (mild degradation)."
        elif avg_trend_slope > 0.05:
            interpretation += f" Average iota improving at +{avg_trend_slope:.3f} per window (performance strengthening)."
    
    return interpretation

def smooth_iotas(iotas, window=3):
    """Apply rolling average smoothing to iota series."""
    if len(iotas) < window:
        return iotas
    
    smoothed = []
    for i in range(len(iotas)):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        smoothed.append(np.mean(iotas[start_idx:end_idx]))
    return smoothed

def plot_individual_metrics_progression(rolling_results: Dict[str, Any], symphony_name: str = "Strategy", save_plot: bool = True, show_plot: bool = False) -> str:
    """Simplified plotting with individual metrics only (no composite iota, no drawdown, no annualized return)."""
    if not HAS_MATPLOTLIB:
        return "Matplotlib not available - install with 'pip install matplotlib' for plots"
    
    if not rolling_results.get('sufficient_data', False):
        return "Insufficient data for plotting"
    
    windows = rolling_results.get('windows', [])
    if len(windows) < 2:
        return "Need at least 2 windows for meaningful plot"
    
    # SAFETY: Limit plotting to prevent crashes
    if len(windows) > 100:
        windows = windows[-100:]  # Use last 100 windows only
        if rolling_results.get('verbose', True):
            print("⚠️  Too many windows - plotting last 100 only")
    
    # Extract data for plotting
    dates = [w['start_date'] for w in windows]
    
    # Extract individual metric iotas (REMOVED ANNUALIZED RETURN AND MAX DRAWDOWN)
    metric_iotas = {}
    for metric in ['sh', 'cr', 'so']:
        metric_iotas[metric] = [w['iotas'][metric] for w in windows]
    
    # Create the plot with only individual metrics
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Add strategy name to the main title
    fig.suptitle(f'{symphony_name} - Individual Metric Iota Progression Over Time', fontsize=16, fontweight='bold')
    
    # Individual metric plots with smoothing (REMOVED ANNUALIZED RETURN AND MAX DRAWDOWN)
    metrics_to_plot = [
        ('sh', 'Sharpe Ratio', 'purple', 's-'),
        ('cr', 'Cumulative Return', 'blue', 'o-'),
        ('so', 'Sortino Ratio', 'orange', 'v-')
    ]
    
    for metric_key, metric_name, color, style in metrics_to_plot:
        # Get valid data points for this metric
        metric_data = []
        metric_dates = []
        for i, (date, iota_val) in enumerate(zip(dates, metric_iotas[metric_key])):
            if np.isfinite(iota_val):
                metric_data.append(iota_val)
                metric_dates.append(date)
        
        if len(metric_data) >= 3:
            # Apply smoothing to individual metrics
            metric_data_smooth = smooth_iotas(metric_data, window=3)
            ax.plot(metric_dates, metric_data_smooth, style, linewidth=2, markersize=4, 
                    color=color, alpha=0.8, label=f'{metric_name} Iota (smoothed)')
            
    
    # Formatting for the plot
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    ax.axhline(y=0.5, color='lightgreen', linestyle=':', alpha=0.3, label='Good Performance (+0.5σ)')
    ax.axhline(y=-0.5, color='lightcoral', linestyle=':', alpha=0.3, label='Poor Performance (-0.5σ)')
    ax.set_ylabel('Iota (ι)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Period (OOS)', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Detection Analysis (Smoothed)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates)//8)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add interpretation text
    risk_level = rolling_results.get('overfitting_risk', 'UNKNOWN')
    avg_trend_slope = rolling_results.get('iota_trend_slope', np.nan)
    n_windows = rolling_results.get('n_windows', 0)
    window_size = rolling_results.get('window_size_days', 0)
    
    risk_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MODERATE': 'gold', 
                   'LOW': 'lightblue', 'MINIMAL': 'lightgreen'}
    risk_color = risk_colors.get(risk_level, 'gray')
    
    interpretation = f"Overfitting Risk: {risk_level} | {n_windows} windows ({window_size}d each) | Smoothed"
    if np.isfinite(avg_trend_slope):
        if avg_trend_slope < -0.1:
            interpretation += f" | Rapid Decline (avg slope: {avg_trend_slope:.3f})"
        elif avg_trend_slope < -0.03:
            interpretation += f" | Declining (avg slope: {avg_trend_slope:.3f})"
        elif avg_trend_slope > 0.05:
            interpretation += f" | Improving (avg slope: +{avg_trend_slope:.3f})"
        else:
            interpretation += f" | Stable (avg slope: {avg_trend_slope:+.3f})"
    
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=risk_color, alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)
    
    # Save the plot with strategy name
    filename = ""
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean strategy name for filename
        clean_name = re.sub(r'[^\w\-_.]', '_', symphony_name)
        filename = f"{clean_name}_metrics_progression_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return filename if save_plot else "Plot generated successfully"

def analyse(url: str, oos_start: str, n_slices: int, overlap: bool, 
           excl: List[Tuple[datetime.date, datetime.date]], verbose: bool = True):
    today = datetime.today().strftime("%Y-%m-%d")
    early = "2000-01-01"

    alloc, sym_name, tickers = fetch_with_progress(url, early, today, verbose)
    
    # Calculate portfolio returns with progress indication
    import contextlib
    import io
    import sys
    
    if verbose:
        if HAS_TQDM:
            # Use progress indication for portfolio calculation
            stop_spinner = threading.Event()
            spinner_thread = threading.Thread(target=spinner_animation, 
                                             args=("Calculating portfolio returns", stop_spinner))
            spinner_thread.daemon = True
            spinner_thread.start()
            
            try:
                # Capture stdout but show progress
                captured_output = io.StringIO()
                with contextlib.redirect_stdout(captured_output):
                    daily_ret, _ = calculate_portfolio_returns(alloc, tickers)
                stop_spinner.set()
                spinner_thread.join(timeout=0.2)
            except Exception as e:
                stop_spinner.set()
                spinner_thread.join(timeout=0.2)
                print("\rCalculating portfolio returns ✗")
                raise e
        else:
            print("Calculating portfolio returns …")
            # Capture and discard stdout during portfolio calculation
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                daily_ret, _ = calculate_portfolio_returns(alloc, tickers)
    else:
        # Silent mode - just capture output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            daily_ret, _ = calculate_portfolio_returns(alloc, tickers)
    
    daily_ret.index = pd.to_datetime(daily_ret.index).date

    # Apply exclusions
    if excl:
        mask = pd.Series(True, index=daily_ret.index)
        for s, e in excl:
            mask &= ~((daily_ret.index >= s) & (daily_ret.index <= e))
        removed = (~mask).sum()
        daily_ret = daily_ret[mask]
        if verbose:
            print(f"Exclusions removed {removed} rows across {len(excl)} window(s).")

    oos_start_dt = datetime.strptime(oos_start, "%Y-%m-%d").date()
    is_ret = daily_ret[daily_ret.index < oos_start_dt]
    oos_ret = daily_ret[daily_ret.index >= oos_start_dt]

    if len(is_ret) < 30 or len(oos_ret) < 30:
        if verbose:
            print("[!] Insufficient data after exclusions (need ≥30 days in each segment).\n")
        return None

    n_oos = len(oos_ret)
    n_is = len(is_ret)

    # Assess statistical reliability
    reliability = assess_sample_reliability(n_is, n_oos)
    
    # ----- OOS metrics (REMOVED MAX DRAWDOWN) -----
    ar_oos = window_cagr(oos_ret)
    sh_oos = sharpe_ratio(oos_ret)
    cr_oos = cumulative_return(oos_ret)
    so_oos = sortino_ratio(oos_ret)

    # ----- build IS slices -----
    slice_len = n_oos
    slices = build_slices(is_ret, slice_len, n_slices, overlap)

    if not slices:
        if verbose:
            print("[!] Could not form any IS slice of required length.\n")
        return None

    # ----- per‑slice metrics (REMOVED MAX DRAWDOWN) -----
    if verbose:
        print(f"Computing metrics for {len(slices)} slices …")
    
    rows = []
    slice_iterator = enumerate(slices[::-1], 1)
    if verbose and len(slices) >= 50 and HAS_TQDM:
        slice_iterator = tqdm(slice_iterator, total=len(slices), desc="Processing slices", ncols=60)
    
    for i, s in slice_iterator:
        rows.append({
            "slice": i,
            "start": s.index[0],
            "end": s.index[-1],
            "ar_is": window_cagr(s),
            "sh_is": sharpe_ratio(s),
            "cr_is": cumulative_return(s),
            "so_is": sortino_ratio(s)
        })
    df = pd.DataFrame(rows)

    # ----- Enhanced statistical analysis with autocorrelation adjustment -----
    if verbose:
        print("Performing statistical analysis with autocorrelation adjustment …")
    
    ar_stats = compute_iota_with_stats(df["ar_is"].values, ar_oos, n_oos, "Annualized Return", verbose=verbose, overlap=overlap)
    sh_stats = compute_iota_with_stats(df["sh_is"].values, sh_oos, n_oos, "Sharpe Ratio", verbose=verbose, overlap=overlap)
    cr_stats = compute_iota_with_stats(df["cr_is"].values, cr_oos, n_oos, "Cumulative Return", verbose=verbose, overlap=overlap)
    so_stats = compute_iota_with_stats(df["so_is"].values, so_oos, n_oos, "Sortino Ratio", verbose=verbose, overlap=overlap)

    # ----- Rolling OOS Analysis for Overfitting Detection (SIMPLIFIED) -----
    if verbose:
        print("Analyzing  …")
    
    rolling_results = rolling_oos_analysis(daily_ret, oos_start_dt, is_ret, n_slices, overlap, verbose=verbose)
    
    results = {
        'symphony': sym_name,
        'reliability': reliability,
        'exclusions': excl,
        'is_window': (is_ret.index[0], is_ret.index[-1], n_is),
        'oos_window': (oos_ret.index[0], oos_ret.index[-1], n_oos),
        'n_slices': len(df),
        'slice_length': slice_len,
        'overlap': overlap,
        'metrics': {
            'ar': ar_stats,
            'sh': sh_stats, 
            'cr': cr_stats,
            'so': so_stats
        },
        'oos_values': {
            'ar': ar_oos,
            'sh': sh_oos,
            'cr': cr_oos,
            'so': so_oos
        },
        'overfitting_analysis': rolling_results
    }

    if not verbose:
        return results

    # ===== ENHANCED OUTPUT WITH AUTOCORRELATION ADJUSTMENT INFO =====
    pc = lambda x: f"{x*100:,.2f}%"
    
    print("========== ENHANCED RESULTS WITH AUTOCORRELATION-ADJUSTED P-VALUES ==========")
    print(f"Symphony          : {sym_name}")
    print(f"Reliability       : {reliability}")
    if excl:
        print("Exclusions        : " + ", ".join([f"{s}→{e}" for s, e in excl]))
    print(f"IS total window   : {is_ret.index[0]} → {is_ret.index[-1]}  ({n_is} days)")
    print(f"OOS window        : {oos_ret.index[0]} → {oos_ret.index[-1]}  ({n_oos} days)")
    print(f"IS slices used    : {len(df)} × {slice_len} days  |  Overlap: {overlap}")
    
    # Show statistical method being used
    test_method = "Autocorrelation-Adjusted Wilcoxon" if overlap else "Standard Wilcoxon"
    bootstrap_method = "Block Bootstrap" if overlap else "Standard Bootstrap"
    print(f"Statistical Test  : {test_method}")
    print(f"Bootstrap Method  : {bootstrap_method}")
    print()

    # Enhanced statistical table with autocorrelation adjustment info
    print("STATISTICAL ANALYSIS:")
    print("=" * 95)
    
    # REMOVED ANNUALIZED RETURN FROM METRICS DATA
    metrics_data = [
        ("Annualized Return", ar_stats, ar_oos, pc),
        ("Sharpe Ratio", sh_stats, sh_oos, lambda x: f"{x:.3f}"),
        ("Cumulative Return", cr_stats, cr_oos, pc),
        ("Sortino Ratio", so_stats, so_oos, format_sortino_output)
    ]
    
    for metric_name, stats_dict, oos_val, formatter in metrics_data:
        print(f"\n{metric_name.upper()}:")
        print(f"  IS Median: {formatter(stats_dict['median_is'])} | OOS: {formatter(oos_val)}")
        
        q25, q75 = stats_dict['iqr_is']
        print(f"  IS Range:  {formatter(q25)} - {formatter(q75)} (25th-75th percentile)")
        
        iota_val = stats_dict['iota']
        print(f"  Iota (ι):  {iota_val:+.3f}")
        print(f"  Direct Interpretation: {interpret_iota_directly(iota_val)}")
        print(f"  Persistence Rating: {stats_dict['persistence_rating']}")
        
        # Enhanced confidence interval display with bootstrap method
        ci_lower, ci_upper = stats_dict['confidence_interval']
        if np.isfinite(ci_lower) and np.isfinite(ci_upper):
            ci_lower_rating = iota_to_persistence_rating(ci_lower)
            ci_upper_rating = iota_to_persistence_rating(ci_upper)
            bootstrap_type = stats_dict.get('bootstrap_method', 'standard_bootstrap')
            bootstrap_display = "Block Bootstrap" if bootstrap_type == 'block_bootstrap' else "Standard Bootstrap"
            print(f"  95% Confidence:    ι: [{ci_lower:+.3f}, {ci_upper:+.3f}] | Rating: [{ci_lower_rating}, {ci_upper_rating}] ({bootstrap_display})")
        
        # Enhanced p-value display with autocorrelation adjustment info
        if np.isfinite(stats_dict['p_value_adjusted']):
            sig_flag = " ***" if stats_dict['significant'] else ""
            test_method_used = stats_dict.get('test_method', 'wilcoxon')
            print(f"  Statistical Test:  p-value = {stats_dict['p_value_adjusted']:.3f}{sig_flag} ({test_method_used})")
            
            # Show autocorrelation adjustment factor if applicable
            if overlap:
                autocorr_adj = stats_dict.get('autocorr_adjustment', 1.0)
                print(f"  Autocorr Adjustment: {autocorr_adj:.3f} (effective sample size factor)")
        
        interpretation = interpret_persistence_rating(stats_dict['persistence_rating'], 
                                                    stats_dict['confidence_interval'])
        print(f"  Rating Interpretation: {interpretation}")

    print("\nRELIABILITY NOTES:")
    if reliability == "INSUFFICIENT_DATA":
        print("⚠️  CRITICAL: Sample sizes too small for reliable analysis")
    elif reliability == "LOW_CONFIDENCE":
        print("⚠️  CAUTION: Limited sample size - results should be interpreted carefully") 
    elif reliability == "MODERATE_CONFIDENCE":
        print("✓  MODERATE: Reasonable sample size for analysis")
    else:
        print("✓  HIGH: Excellent sample size for reliable analysis")
    
    print("\nINTUITIVE IOTA INTERPRETATION:")
    print("Iota (ι) measures how many standard deviations OOS performance differs from IS median.")
    print("It's calculated as: ι = weight × (OOS - IS_median) / IS_std_dev")
    print("• Positive ι = Better OOS performance (strategy improved)")
    print("• Negative ι = Worse OOS performance (strategy degraded)")  
    print("• |ι| ≥ 1.0 = Major difference (>1 standard deviation)")
    print("• |ι| < 0.1 = Minimal difference (within noise)")
    print("\nPERSISTENCE RATING SCALE:")
    print("100 = Neutral (OOS matches IS), >100 = Outperformance, <100 = Underperformance")
    print("*** = Statistically significant difference (p < 0.05, autocorrelation-adjusted)")
    
    # ----- Overfitting Risk Analysis -----
    print("\n" + "=" * 95)
    print("OVERFITTING RISK ANALYSIS:")
    overfitting_interp = interpret_overfitting_risk(rolling_results)
    print(f"• {overfitting_interp}")
    
    if rolling_results.get('sufficient_data', False):
        print(f"• Analysis based on {rolling_results['n_windows']} rolling windows of {rolling_results['window_size_days']} days each")
        print(f"• Each window compared against {rolling_results['is_slices_used']} historical slices")
        
        if np.isfinite(rolling_results.get('iota_trend_slope', np.nan)):
            slope_direction = "improving" if rolling_results['iota_trend_slope'] > 0 else "declining"
            print(f"• Average iota trend: {slope_direction} at {rolling_results['iota_trend_slope']:+.4f} per window")
        
        # Warning for young strategies
        total_oos_days = len(oos_ret)
        if total_oos_days < 252:
            print(f"⚠️  SHORT OOS PERIOD: Only {total_oos_days} days OOS (recommend ≥252 days)")
        
        # Performance degradation warning
        if rolling_results['overfitting_risk'] in ['MODERATE', 'HIGH', 'CRITICAL']:
            print("⚠️  RECOMMENDATION: Strategy may be overfit and/or market environment has changed - extend backtest if you can")
        
        # Additional insights
        if rolling_results.get('degradation_score', 0) >= 3:
            print("⚠️  PATTERN: Strategy performance increasingly deviates from backtest expectations")
        
        # Generate and save plot
        if HAS_MATPLOTLIB:
            plot_filename = plot_individual_metrics_progression(rolling_results, sym_name, save_plot=True, show_plot=False)
            if plot_filename.endswith('.png'):
                print(f"📊 Individual metrics progression plot saved: {plot_filename}")
            else:
                print(f"📊 Plot status: {plot_filename}")
        else:
            print("📊 Install matplotlib for individual metrics progression plots: pip install matplotlib")
    
    print("=" * 95)

    # ===== ENHANCED CSV LOGGING WITH AUTOCORRELATION DETAILS =====
    out = Path.cwd() / "iota_log_autocorr_adjusted_v1.csv"
    
    # Helper function to safely get values, handling NaN cases
    safe_get = lambda d, k, default=np.nan: d.get(k, default) if d.get(k) is not None else default
    
    log_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "symphony": sym_name,
        "url": url,
        "oos_start": oos_start,
        "n_slices": len(df),
        "overlap": overlap,
        "oos_days": n_oos,
        "is_days": n_is,
        "reliability": reliability,
        
        # Enhanced metrics with autocorrelation-adjusted p-values
        "med_ar_is": safe_get(ar_stats, 'median_is'),
        "ar_oos": ar_oos,
        "iota_ar": safe_get(ar_stats, 'iota'),
        "rating_ar": safe_get(ar_stats, 'persistence_rating'),
        "ar_ci_lower": safe_get(ar_stats, 'confidence_interval', (np.nan, np.nan))[0],
        "ar_ci_upper": safe_get(ar_stats, 'confidence_interval', (np.nan, np.nan))[1],
        "ar_p_value_adj": safe_get(ar_stats, 'p_value_adjusted'),
        "ar_significant": safe_get(ar_stats, 'significant', False),
        "ar_autocorr_adj": safe_get(ar_stats, 'autocorr_adjustment', 1.0),
        "ar_test_method": safe_get(ar_stats, 'test_method', 'unknown'),
        "ar_bootstrap_method": safe_get(ar_stats, 'bootstrap_method', 'unknown'),
        
        "med_sh_is": safe_get(sh_stats, 'median_is'),
        "sh_oos": sh_oos,
        "iota_sh": safe_get(sh_stats, 'iota'),
        "rating_sh": safe_get(sh_stats, 'persistence_rating'),
        "sh_ci_lower": safe_get(sh_stats, 'confidence_interval', (np.nan, np.nan))[0],
        "sh_ci_upper": safe_get(sh_stats, 'confidence_interval', (np.nan, np.nan))[1],
        "sh_p_value_adj": safe_get(sh_stats, 'p_value_adjusted'),
        "sh_significant": safe_get(sh_stats, 'significant', False),
        "sh_autocorr_adj": safe_get(sh_stats, 'autocorr_adjustment', 1.0),
        "sh_test_method": safe_get(sh_stats, 'test_method', 'unknown'),
        "sh_bootstrap_method": safe_get(sh_stats, 'bootstrap_method', 'unknown'),
        
        "med_cr_is": safe_get(cr_stats, 'median_is'),
        "cr_oos": cr_oos,
        "iota_cr": safe_get(cr_stats, 'iota'),
        "rating_cr": safe_get(cr_stats, 'persistence_rating'),
        "cr_ci_lower": safe_get(cr_stats, 'confidence_interval', (np.nan, np.nan))[0],
        "cr_ci_upper": safe_get(cr_stats, 'confidence_interval', (np.nan, np.nan))[1],
        "cr_p_value_adj": safe_get(cr_stats, 'p_value_adjusted'),
        "cr_significant": safe_get(cr_stats, 'significant', False),
        "cr_autocorr_adj": safe_get(cr_stats, 'autocorr_adjustment', 1.0),
        "cr_test_method": safe_get(cr_stats, 'test_method', 'unknown'),
        "cr_bootstrap_method": safe_get(cr_stats, 'bootstrap_method', 'unknown'),
        
        "med_so_is": safe_get(so_stats, 'median_is'),
        "so_oos": so_oos,
        "iota_so": safe_get(so_stats, 'iota'),
        "rating_so": safe_get(so_stats, 'persistence_rating'),
        "so_ci_lower": safe_get(so_stats, 'confidence_interval', (np.nan, np.nan))[0],
        "so_ci_upper": safe_get(so_stats, 'confidence_interval', (np.nan, np.nan))[1],
        "so_p_value_adj": safe_get(so_stats, 'p_value_adjusted'),
        "so_significant": safe_get(so_stats, 'significant', False),
        "so_autocorr_adj": safe_get(so_stats, 'autocorr_adjustment', 1.0),
        "so_test_method": safe_get(so_stats, 'test_method', 'unknown'),
        "so_bootstrap_method": safe_get(so_stats, 'bootstrap_method', 'unknown'),
        
        # Simplified overfitting analysis (removed composite iota and max drawdown)
        "overfitting_risk": rolling_results.get('overfitting_risk', 'UNKNOWN'),
        "n_rolling_windows": rolling_results.get('n_windows', 0),
        "avg_iota_trend_slope": rolling_results.get('iota_trend_slope', np.nan),
        "degradation_score": rolling_results.get('degradation_score', np.nan),
        "window_size_days": rolling_results.get('window_size_days', np.nan),
        "step_size_days": rolling_results.get('step_size_days', np.nan),
        "total_oos_days": rolling_results.get('total_oos_days', len(oos_ret)),
        
        # Individual metric slopes for detailed analysis (removed max drawdown only)
        "ar_slope": rolling_results.get('metric_slopes', {}).get('ar_slope', np.nan),
        "sh_slope": rolling_results.get('metric_slopes', {}).get('sh_slope', np.nan),
        "cr_slope": rolling_results.get('metric_slopes', {}).get('cr_slope', np.nan),
        "so_slope": rolling_results.get('metric_slopes', {}).get('so_slope', np.nan),
        
        # Confidence interval widths (uncertainty measures) - removed max drawdown only
        "ar_ci_width": abs(safe_get(ar_stats, 'confidence_interval', (np.nan, np.nan))[1] - safe_get(ar_stats, 'confidence_interval', (np.nan, np.nan))[0]),
        "sh_ci_width": abs(safe_get(sh_stats, 'confidence_interval', (np.nan, np.nan))[1] - safe_get(sh_stats, 'confidence_interval', (np.nan, np.nan))[0]),
        "cr_ci_width": abs(safe_get(cr_stats, 'confidence_interval', (np.nan, np.nan))[1] - safe_get(cr_stats, 'confidence_interval', (np.nan, np.nan))[0]),
        "so_ci_width": abs(safe_get(so_stats, 'confidence_interval', (np.nan, np.nan))[1] - safe_get(so_stats, 'confidence_interval', (np.nan, np.nan))[0]),
        
        "exclusions": ";".join([f"{s}:{e}" for s, e in excl]) if excl else ""
    }
    
    # Write to CSV with error handling
    try:
        pd.DataFrame([log_data]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"\nAutocorrelation-adjusted results appended to {out}")
    except Exception as e:
        print(f"\nWarning: Could not write to CSV log: {e}")
    
    # Generate strategy summary report
    summary_filename = generate_strategy_summary(results, url)
    if summary_filename and summary_filename.endswith('.txt'):
        print(f"\n📄 Strategy summary saved to: {summary_filename}")
    
    # Generate enhanced README file
    generate_readme()
    
    print()

    return results

def generate_strategy_summary(results: Dict[str, Any], url: str) -> str:
    """Generate a concise summary report for the strategy analysis."""
    if not results:
        print("No results to summarize")
        return "No results to summarize"
        
    symphony_name = results.get('symphony', 'Unknown_Strategy')
    print(f"Symphony name: {symphony_name}")
    
    # Clean strategy name for filename
    clean_name = re.sub(r'[^\w\-_.]', '_', symphony_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_name}_Iota_Summary_{timestamp}.txt"
    
    print(f"Attempting to create file: {filename}")
    
    # Helper functions for formatting
    pc = lambda x: f"{x*100:,.2f}%"
    
    # Extract key metrics with error checking
    try:
        ar_stats = results['metrics']['ar']
        sh_stats = results['metrics']['sh'] 
        cr_stats = results['metrics']['cr']
        so_stats = results['metrics']['so']
        
        ar_oos = results['oos_values']['ar']
        sh_oos = results['oos_values']['sh']
        cr_oos = results['oos_values']['cr']
        so_oos = results['oos_values']['so']
        
        overfitting_analysis = results['overfitting_analysis']
        
        print("Successfully extracted all metrics")
    except KeyError as e:
        print(f"DEBUG: Missing key in results: {e}")
        return f"Error: Missing data in results: {e}"
    
    # Generate summary content
    summary_content = f"""
{symphony_name} - IOTA ANALYSIS SUMMARY
{'=' * (len(symphony_name) + 25)}

ANALYSIS DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
STRATEGY URL: {url}

ANALYSIS OVERVIEW
-----------------
• Reliability Level: {results['reliability']}
• IS Period: {results['is_window'][0]} to {results['is_window'][1]} ({results['is_window'][2]} days)
• OOS Period: {results['oos_window'][0]} to {results['oos_window'][1]} ({results['oos_window'][2]} days)
• Analysis Method: {results['n_slices']} slices, {'Overlapping' if results['overlap'] else 'Non-overlapping'}

PERFORMANCE SUMMARY
-------------------
                    IS Median    OOS Actual    Iota (ι)    Rating    Significance
Annualized Return   {pc(ar_stats['median_is']):>10}   {pc(ar_oos):>10}   {ar_stats['iota']:>+7.3f}   {ar_stats['persistence_rating']:>6}   {'***' if ar_stats['significant'] else '   '}
Sharpe Ratio        {sh_stats['median_is']:>10.3f}   {sh_oos:>10.3f}   {sh_stats['iota']:>+7.3f}   {sh_stats['persistence_rating']:>6}   {'***' if sh_stats['significant'] else '   '}
Cumulative Return   {pc(cr_stats['median_is']):>10}   {pc(cr_oos):>10}   {cr_stats['iota']:>+7.3f}   {cr_stats['persistence_rating']:>6}   {'***' if cr_stats['significant'] else '   '}
Sortino Ratio       {so_stats['median_is']:>10.3f}   {so_oos:>10.3f}   {so_stats['iota']:>+7.3f}   {so_stats['persistence_rating']:>6}   {'***' if so_stats['significant'] else '   '}

KEY INSIGHTS
------------
"""

    # Add key insights based on iota values and significance
    insights = []
    
    # Overall performance assessment
    avg_iota = np.mean([ar_stats['iota'], sh_stats['iota'], cr_stats['iota'], so_stats['iota']])
    if avg_iota >= 0.5:
        insights.append("✓ STRONG: Strategy showing consistent outperformance vs. historical expectations")
    elif avg_iota >= 0.1:
        insights.append("✓ POSITIVE: Strategy performing moderately better than historical median")
    elif avg_iota >= -0.1:
        insights.append("→ NEUTRAL: Strategy performance aligns with historical expectations")
    elif avg_iota >= -0.5:
        insights.append("⚠ CAUTION: Strategy underperforming relative to historical median")
    else:
        insights.append("⚠ CONCERN: Strategy significantly underperforming historical expectations")
    
    # Significance assessment
    significant_metrics = sum([ar_stats['significant'], sh_stats['significant'], 
                              cr_stats['significant'], so_stats['significant']])
    if significant_metrics >= 3:
        insights.append("✓ HIGH CONFIDENCE: Multiple metrics show statistically significant differences")
    elif significant_metrics >= 1:
        insights.append("→ MODERATE CONFIDENCE: Some metrics show statistical significance")
    else:
        insights.append("⚠ LOW CONFIDENCE: No metrics show statistical significance")
    
    # Risk-adjusted performance
    risk_adjusted_avg = np.mean([sh_stats['iota'], so_stats['iota']])
    if risk_adjusted_avg >= 0.3:
        insights.append("✓ RISK-ADJUSTED: Strong risk-adjusted performance (Sharpe/Sortino)")
    elif risk_adjusted_avg <= -0.3:
        insights.append("⚠ RISK-ADJUSTED: Weak risk-adjusted performance")
    
    # Overfitting assessment
    if overfitting_analysis.get('sufficient_data', False):
        risk_level = overfitting_analysis['overfitting_risk']
        if risk_level in ['MINIMAL', 'LOW']:
            insights.append("✓ OVERFITTING: Low overfitting risk detected")
        elif risk_level == 'MODERATE':
            insights.append("⚠ OVERFITTING: Moderate overfitting risk - monitor closely")
        else:
            insights.append("⚠ OVERFITTING: High overfitting risk - strategy may be curve-fit")
    
    # Add insights to summary
    for insight in insights:
        summary_content += f"• {insight}\n"
    
    # Add overfitting analysis details
    if overfitting_analysis.get('sufficient_data', False):
        summary_content += f"""
OVERFITTING ANALYSIS
--------------------
• Risk Level: {overfitting_analysis['overfitting_risk']}
• Rolling Windows: {overfitting_analysis['n_windows']} windows of {overfitting_analysis['window_size_days']} days each
• Trend Analysis: {'Improving' if overfitting_analysis.get('iota_trend_slope', 0) > 0 else 'Declining'} at {overfitting_analysis.get('iota_trend_slope', 0):+.4f} per window
• Total OOS Days: {overfitting_analysis['total_oos_days']}
"""
    else:
        summary_content += """
OVERFITTING ANALYSIS
--------------------
• Status: Insufficient data for overfitting analysis
• Recommendation: Allow OOS period to extend to at least 6 months for meaningful analysis
"""

    # Add interpretation guide
    summary_content += """
INTERPRETATION GUIDE
--------------------
• IOTA (ι): Measures standard deviations from historical median
  - Positive = Better than expected | Negative = Worse than expected
  - |ι| ≥ 1.0 = Major difference | |ι| < 0.1 = Minimal difference

• PERSISTENCE RATING: 0-500 scale converting iota to intuitive score
  - 100 = Neutral | >100 = Outperformance | <100 = Underperformance

• SIGNIFICANCE (***): P-value < 0.05 (autocorrelation-adjusted)
  - Indicates statistically robust difference from historical expectations

RECOMMENDATIONS
---------------"""

    # Add specific recommendations
    recommendations = []
    
    if results['reliability'] in ['INSUFFICIENT_DATA', 'LOW_CONFIDENCE']:
        recommendations.append("• Extend analysis period to improve statistical reliability")
    
    if overfitting_analysis.get('overfitting_risk') in ['HIGH', 'CRITICAL']:
        recommendations.append("• Consider longer backtesting - strategy may be overfit to limited or cherry-picked historical data")
        recommendations.append("• Implement more conservative position sizing")
    
    if avg_iota < -0.5:
        recommendations.append("• Review strategy logic - consistent underperformance detected")
        recommendations.append("• Consider market regime changes or parameter adjustments")
    
    if significant_metrics == 0:
        recommendations.append("• Results may be due to random variation - extend OOS period")
    
    if len(recommendations) == 0:
        recommendations.append("• Strategy appears to be performing within expected parameters")
        recommendations.append("• Continue monitoring with periodic re-analysis")
    
    for rec in recommendations:
        summary_content += f"{rec}\n"
    
    summary_content += f"""
TECHNICAL DETAILS
-----------------
• Analysis Tool: Iota Calculator v1.0 | Gobi
• Statistical Method: {'Overlapping' if results['overlap'] else 'Non-overlapping'} slice analysis
• Bootstrap Method: {'Block Bootstrap' if results['overlap'] else 'Standard Bootstrap'}
• Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

For detailed analysis, see the full terminal output and CSV log files.
For questions about methodology, refer to IOTA_CALCULATOR_README.txt
"""

    # Write summary to file
    try:
        with open(filename, 'w') as f:
            f.write(summary_content.strip())
        return filename
    except Exception as e:
        print(f"Warning: Could not write summary file: {e}")
        return f"Summary generated but not saved: {e}"
    
def generate_readme():    
    """Generate a comprehensive README file explaining the autocorrelation-adjusted iota calculator."""
    readme_path = Path.cwd() / "IOTA_CALCULATOR_README.txt"
    
    # Only generate if it doesn't exist to avoid overwriting user modifications
    if readme_path.exists():
        return
    
    readme_content = """
# IOTA CALCULATOR | OUT-OF-SAMPLE PERFORMANCE QUANTIFIER
## README & USER GUIDE (SIMPLIFIED VERSION)

================================================================================

## OVERVIEW
The Iota Calculator evaluates how well your trading strategy's backtest results are translating to out-of-sample (OOS) performance. 

This is a RETROSPECTIVE analysis tool - it does not predict future performance, but rather helps you understand whether your OOS results indicate the strategy is performing as expected based on historical patterns.

## DETAILED METHODOLOGY

### Step 1: Data Preparation and Slice Construction
**What happens:**
1. Historical data is split into In-Sample (IS) and Out-of-Sample (OOS) periods at your specified date
2. The IS period is divided into multiple overlapping or non-overlapping "slices"
3. Each slice has the same length as your OOS period for fair comparison

**Rationale:**
- **Temporal consistency**: Each IS slice represents what your strategy would have done during a period of identical length to your actual OOS period
- **Distribution building**: Multiple slices create a distribution of historical performance under similar conditions
- **Avoiding look-ahead bias**: Only data prior to OOS start date is used for IS analysis

**Example:**
- OOS period: 2023-01-01 to 2024-01-01 (365 days)
- IS slices: 100 overlapping 365-day periods from historical data before 2023-01-01
- Each slice represents "what if the strategy had run for 365 days during this historical period"

### Step 2: Metric Calculation
**What happens:**
1. Three core metrics calculated for OOS period: Sharpe Ratio, Cumulative Return, Sortino Ratio
2. Same three metrics calculated for each IS slice
3. Statistical distribution properties computed for IS metrics (median, standard deviation, quartiles)

**Rationale:**
- **Focus on performance**: These metrics capture different aspects of strategy performance without redundancy
- **Risk-adjustment**: Sharpe and Sortino ratios account for volatility and downside risk, but Sortino is preferable when evaluating high volatility strategies  
- **Comparability**: Same metrics across all time periods enable direct comparison

### Step 3: Iota Calculation
**What happens:**
```
ι = weight × (OOS_metric - IS_median) / IS_std_dev
```

Where:
- `weight = min(1.0, √(OOS_days / 252))` - sample size adjustment
- `IS_median` - median of the IS slice distribution
- `IS_std_dev` - standard deviation of IS slice distribution

**Rationale:**
- **Standardization**: Converts absolute differences to standard deviation units for universal interpretation
- **Sample size weighting**: Longer OOS periods get more weight (up to 1 year = full weight)
- **Robust statistics**: Median and standard deviation are less sensitive to outliers than mean
- **Intuitive scale**: ι = +1.0 means OOS performed 1 standard deviation better than typical

### Step 4: Statistical Testing with Autocorrelation Adjustment
**What happens:**
1. **For overlapping slices**: Calculate first-order autocorrelation of IS values
2. **Effective sample size**: Adjust for temporal correlation using Newey-West correction
3. **P-value adjustment**: Scale p-values upward to account for reduced independence
4. **Bootstrap confidence intervals**: Use block bootstrap for overlapping data, standard bootstrap for non-overlapping

**Rationale:**
- **Overlapping problem**: Adjacent overlapping slices share most of their data, violating independence assumptions
- **Conservative testing**: Better to be cautious about statistical significance than overconfident
- **Temporal structure**: Block bootstrap preserves the time-series properties of financial data

**Mathematical detail:**
```
effective_n = n × (1 - autocorr) / (1 + autocorr)
adjustment_factor = √(effective_n / n)
p_value_adjusted = min(1.0, p_value_raw / adjustment_factor)
```

### Step 5: Rolling Window Analysis (Overfitting Detection)
**What happens:**
1. **Window creation**: OOS period divided into overlapping windows (e.g., 6-month windows with 1-month steps)
2. **Historical comparison**: Each window compared against IS slice distribution
3. **Trend analysis**: Linear regression on iota values over time
4. **Degradation scoring**: Multiple criteria assess performance decay

**Rationale:**
- **Overfitting detection**: Strategies that are overfit show declining performance over time
- **Temporal granularity**: Rolling windows reveal when and how performance changes
- **Early warning**: Identifies degradation before it becomes severe

## INTERPRETATION GUIDE

### Understanding Iota Values
- **ι = +2.0**: Exceptional performance (>2 standard deviations above historical median)
- **ι = +1.0**: Excellent performance (1 standard deviation above median)
- **ι = +0.5**: Good performance (0.5 standard deviations above median)
- **ι = 0.0**: Neutral performance (matches historical median exactly)
- **ι = -0.5**: Caution warranted (0.5 standard deviations below median)
- **ι = -1.0**: Poor performance (1 standard deviation below median)
- **ι = -2.0**: Critical underperformance (>2 standard deviations below median)


### Persistence Ratings (Expanded Explanation)

The **Persistence Rating** converts iota (ι) — a standardized measure of OOS deviation from historical expectations — into a more intuitive 0–500 point scale.

Persistence Rating = 100 × exp(0.5 × ι)


#### 🧠 Interpretation:

| Iota (ι) | Rating | Meaning |
|----------|--------|---------|
| +2.0     | ~270   | **Exceptional** outperformance vs. history |
| +1.0     | ~165   | **Excellent** — OOS is ~1σ above IS median |
| +0.5     | ~128   | **Good** — modest but real OOS improvement |
|  0.0     | 100    | **Neutral** — matches historical median |
| –0.5     | ~78    | **Caution** — mild underperformance |
| –1.0     | ~60    | **Poor** — significant degradation |
| –2.0     | ~36    | **Critical** — >2σ below historical norm |

- **>100** = Outperformance relative to in-sample history
- **<100** = Underperformance
- **≈100** = Performance similar to backtest expectations

#### 🎯 Why Use a Rating?

Persistence Rating offers a **non-technical summary of iota** that helps you quickly judge whether your strategy is holding up OOS:
- Compresses a wide range of iota values into a bounded and interpretable scale
- Makes cross-strategy comparisons easier — e.g., Rating 170 vs. Rating 90
- Designed for intuitive "traffic light"-style interpretation:
  - **>130** = Signals improved OOS
  - **90–110** = Neutral - Behaving as Expected
  - **<80** = Warning signs - Degradation OOS

#### ⚖️ Important Notes:
- Rating is **not a p-value** — it doesn't reflect statistical significance alone
- Ratings are more meaningful when **confidence intervals are narrow**
- Use alongside adjusted p-values and degradation trends for full context


### Statistical Significance and P-Values

#### **What the P-Value Means**
The p-value answers the question: "If my strategy actually performed no differently than random historical periods, what's the probability I would see a difference this large or larger by pure chance?"

**Example interpretations:**
- **p = 0.001**: Only 0.1% chance this difference is due to random luck
- **p = 0.050**: 5% chance this difference is due to random luck  
- **p = 0.200**: 20% chance this difference is due to random luck

#### **Why P-Values Often Show 0.000**
- **Display rounding**: Values like 0.0003 display as "0.000" (rounded to 3 decimal places)
- **Very strong effects**: Your strategy may genuinely differ dramatically from historical expectations
- **Conservative testing**: After autocorrelation adjustment, even small raw p-values indicate robust significance

#### **Statistical Significance Markers**
- ***** (3 asterisks) = p < 0.05 after autocorrelation adjustment = "statistically significant"
- **No asterisks**: p ≥ 0.05 = difference could plausibly be due to random variation

#### **Autocorrelation Adjustment Impact**
When you see "Autocorr Adjustment: 0.126", this means:
- Your overlapping slices have **strong temporal correlation**
- The effective sample size is only **12.6%** of the nominal sample size
- P-values are adjusted **upward** (made more conservative) to account for this
- **Lower adjustment factors** = **stronger correlation** = **more conservative testing**

**Common adjustment factor ranges:**
- **1.000**: No overlap, no adjustment needed
- **0.700**: Moderate overlap, typical for financial data
- **0.300**: Heavy overlap, very conservative adjustment
- **0.126**: Extreme overlap, maximally conservative testing

#### **What This Means for Your Strategy**
If your p-value is very small (displays as 0.000) even after a strong autocorrelation adjustment:
1. **High confidence**: The performance difference is very unlikely due to chance
2. **Robust result**: Survives conservative statistical testing
3. **Strong signal**: Your strategy genuinely differs from historical expectations

#### **Confidence Intervals**
- **95% range** of plausible iota values accounting for uncertainty
- **Narrow intervals**: High precision, confident in the estimate
- **Wide intervals**: High uncertainty, need more data or longer periods
- **Intervals crossing zero**: Performance difference might not be meaningful

## ROLLING IOTA GRAPHICAL OUTPUT INTERPRETATION

### What the Plot Shows
The rolling iota plot displays how your strategy's performance consistency changes over time during the out-of-sample period.

### Key Elements to Look For:


#### 1. **Reference Horizontal Lines**
- **Gray line at ι = 0**: Neutral performance (matches historical median)
- **Green dotted line at ι = +0.5**: Good performance threshold
- **Red dotted line at ι = -0.5**: Poor performance threshold

#### 2. **Individual Metric Lines**
- **Purple (Sharpe Ratio)**: Risk-adjusted performance trend
- **Blue (Cumulative Return)**: Total return accumulation trend
- **Orange (Sortino Ratio)**: Downside risk-adjusted performance trend

#### 3. **Smoothing (3-period moving average)**
- Reduces noise to show clearer underlying trends
- Helps identify systematic patterns vs. random fluctuations

### Diagnostic Patterns:

#### **Healthy Pattern (Low Overfitting Risk)**
- Iotas fluctuate around zero with no strong downward trend
- Multiple metrics show similar, stable patterns
- Trend slopes near zero or slightly positive

#### **Warning Pattern (Moderate Risk)**
- Gradual decline in iotas over time
- Some metrics declining while others stable
- Trend slopes between -0.05 and -0.15

#### **Critical Pattern (High Overfitting Risk)**
- Sharp downward trends in multiple metrics
- Iotas starting positive but ending negative
- Trend slopes below -0.15
- Wide divergence between different metrics

### Bottom Text Box Information:
- **Overfitting Risk Level**: MINIMAL/LOW/MODERATE/HIGH/CRITICAL
- **Number of windows**: How many time periods analyzed
- **Window size**: Length of each rolling period (e.g., 126 days = ~6 months)
- **Trend slope**: Average rate of iota change per window
- **"Smoothed" indicator**: Confirms 3-period moving average applied

### Actionable Insights:

#### **If you see declining trends:**
1. **Moderate decline**: Monitor closely, consider shorter rebalancing periods
2. **Steep decline**: Strategy likely overfit, consider re-optimization
3. **Mixed signals**: Some metrics declining, others stable - investigate which aspects are failing

#### **If you see stable/improving trends:**
1. **Flat trends**: Strategy performing as expected
2. **Improving trends**: Strategy may be conservative in backtest, performing better in reality
3. **Highly volatile**: Consider longer evaluation periods or different metrics

### Common Misinterpretations to Avoid:
- **Short-term noise**: Don't overreact to single bad windows
- **Seasonal effects**: Some decline might be due to market regime changes, not overfitting
- **Insufficient data**: Need at least 6 windows for meaningful trend analysis

## WHAT IS IOTA (ι)?
Iota is a standardized metric that measures how many standard deviations your out-of-sample performance differs from the in-sample median, adjusted for sample size.

Formula: ι = weight × (OOS_metric - IS_median) / IS_std_dev

Where:
- weight = min(1.0, √(OOS_days / 252)) accounts for sample size reliability
- OOS_metric = your strategy's out-of-sample performance value
- IS_median = median of all in-sample slice performances  
- IS_std_dev = standard deviation of in-sample slice performances

INTERPRETATION:
- ι = +1.0: OOS performed 1 standard deviation BETTER than historical median
- ι = -1.0: OOS performed 1 standard deviation WORSE than historical median
- ι = 0: OOS performance matches historical expectations exactly
- |ι| ≥ 1.0: Major difference (statistically significant)
- |ι| < 0.1: Minimal difference (within noise)

## AUTOCORRELATION ADJUSTMENT
When using overlapping slices, the temporal correlation between adjacent slices reduces the effective sample size and can lead to overly optimistic p-values.

This calculator automatically:
1. Detects overlapping slice configurations
2. Calculates the first-order autocorrelation of IS slice metrics
3. Adjusts the effective sample size using Newey-West type correction
4. Provides more conservative, statistically valid p-values

The adjustment factor is reported for transparency, typically ranging from 0.3-1.0:
- 1.0 = No adjustment (non-overlapping slices)
- 0.7 = Moderate positive autocorrelation typical of overlapping financial data
- 0.3 = High positive autocorrelation requiring significant adjustment

## ENHANCED BOOTSTRAP METHODS
The calculator uses overlap-aware bootstrap methods for accurate confidence intervals:

**OVERLAPPING SLICES:**
- Uses Block Bootstrap to preserve temporal structure
- Accounts for autocorrelation in overlapping time series data
- More robust confidence intervals for dependent data

**NON-OVERLAPPING SLICES:**
- Uses Standard Bootstrap for independent samples
- Traditional resampling for clean statistical independence

## CORE METRICS ANALYZED
This simplified version focuses on three key performance metrics:

1. **SHARPE RATIO**: Risk-adjusted return measure
2. **CUMULATIVE RETURN**: Total return over period
3. **SORTINO RATIO**: Downside risk-adjusted return

Note: Annualized return has been removed to avoid calculation inflation issues.

## OVERFITTING DETECTION
The calculator includes rolling analysis to detect overfitting patterns:

**ROLLING WINDOW ANALYSIS:**
Creates multiple overlapping out-of-sample windows and tracks how each metric's iota changes over time. Declining trends suggest the strategy is increasingly deviating from backtest expectations.

**OVERFITTING RISK LEVELS:**
- MINIMAL: Stable performance relative to backtest
- LOW: Minor inconsistencies
- MODERATE: Some degradation detected
- HIGH: Significant degradation - likely overfit
- CRITICAL: Severe degradation - high confidence of overfitting

**VISUAL ANALYSIS:**
When matplotlib is installed, generates plots showing individual metric iota progression over time with trend lines and risk assessment.

## ENHANCED LOGGING
The CSV output includes:
- Autocorrelation adjustment factors for each metric
- Adjusted p-values accounting for temporal correlation
- Bootstrap method used (block vs standard)
- Individual metric trend slopes
- Confidence interval widths as uncertainty measures

## BEST PRACTICES
1. Use OOS periods of at least 6 months for meaningful results
2. Focus on risk-adjusted metrics (Sharpe, Sortino) over raw returns
3. Monitor overfitting risk, especially for strategies <2 years old
4. Trust autocorrelation-adjusted p-values for overlapping slice analysis
5. Consider confidence interval width as measure of uncertainty
6. Be skeptical of strategies showing declining iota trends over time

## STATISTICAL IMPROVEMENTS
This version provides:
- Autocorrelation-adjusted p-values for overlapping slice temporal correlation
- More conservative and statistically valid hypothesis tests
- Block bootstrap confidence intervals preserving temporal structure
- Simplified analysis focusing on core performance metrics
- Enhanced uncertainty quantification

## FILE OUTPUTS
- iota_log_autocorr_adjusted_v1.csv: Results with autocorrelation adjustments
- individual_metrics_progression_YYYYMMDD_HHMMSS.png: Individual metric plots
- IOTA_CALCULATOR_AUTOCORR_README.txt: This documentation

================================================================================
For questions or issues, ask an LLM or reach out to @gobi on Discord.
================================================================================
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content.strip())
    
    print(f"Documentation saved to {readme_path}")

def run_batch_analysis(config_file: str, verbose: bool = True) -> List[Dict]:
    """Run batch analysis from configuration file."""
    import json
    
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
    except Exception as e:
        print(f"[!] Error reading config file: {e}")
        return []
    
    results = []
    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"\n{'='*20} BATCH ANALYSIS {i}/{len(configs)} {'='*20}")
        
        try:
            url = config['url']
            oos_start = config['oos_start']
            n_slices = config.get('n_slices', 100)
            overlap = config.get('overlap', True)
            exclusions_str = config.get('exclusions', '')
            exclusions = parse_exclusion_input(exclusions_str)
            
            result = analyse(url, oos_start, n_slices, overlap, exclusions, verbose)
            if result:
                results.append(result)
                
        except Exception as e:
            if verbose:
                print(f"[!] Error in batch item {i}: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Iota Calculator"
    )
    parser.add_argument("--overlap", dest="overlap", action="store_true", 
                       help="allow overlapping IS slices (default)")
    parser.add_argument("--no-overlap", dest="overlap", action="store_false", 
                       help="use non‑overlapping IS slices")
    parser.add_argument("--batch", type=str, metavar="CONFIG_FILE",
                       help="run batch analysis from JSON config file")
    parser.add_argument("--run-once", action="store_true",
                       help="run single analysis and exit (for scripting)")
    parser.add_argument("--url", type=str, help="Composer symphony URL")
    parser.add_argument("--oos-start", type=str, help="OOS start date (YYYY-MM-DD)")
    parser.add_argument("--n-slices", type=int, default=100, help="number of IS slices")
    parser.add_argument("--exclusions", type=str, default="", 
                       help="exclusion windows (comma-separated)")
    parser.add_argument("--quiet", action="store_true", help="suppress output (return results only)")
    
    parser.set_defaults(overlap=True)
    args = parser.parse_args()

    if not args.quiet:
        print("\nGobi's Iota Calculator")

    # Batch mode
    if args.batch:
        results = run_batch_analysis(args.batch, verbose=not args.quiet)
        if args.quiet:
            return results
        else:
            print(f"\nBatch analysis complete. Processed {len(results)} configurations.")
            return results

    # Single run mode (scripting)
    if args.run_once:
        if not args.url or not args.oos_start:
            print("[!] --run-once requires --url and --oos-start")
            return None
        
        exclusions = parse_exclusion_input(args.exclusions)
        result = analyse(args.url, args.oos_start, args.n_slices, args.overlap, 
                        exclusions, verbose=not args.quiet)
        return result

    # Interactive mode (original behavior)
    default_oos = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    while True:
        url = input("Enter Composer symphony URL (or <Enter> to quit): ").strip()
        if not url:
            print("Goodbye ✌️")
            break

        oos = input(f"Enter OOS start [YYYY‑MM‑DD] (default {default_oos}): ").strip() or default_oos
        try:
            datetime.strptime(oos, "%Y-%m-%d")
        except ValueError:
            print("[!] Bad date format. Try again.\n")
            continue

        slices_str = input("Enter # IS slices (default 100): ").strip() or "100"
        try:
            slices_n = max(1, int(slices_str))
        except ValueError:
            print("[!] Invalid integer.\n")
            continue

        excl_str = input("Exclude windows? (e.g., 2020-03-01 to 2020-05-01, 2022-01-01 to 2022-02-01): ")
        exclusions = parse_exclusion_input(excl_str)

        try:
            analyse(url, oos, slices_n, args.overlap, exclusions)
        except Exception as e:
            print(f"[!] Error: {e}\n")

def compare_strategies(results_list: List[Dict], metric: str = 'sh') -> pd.DataFrame:
    """Compare multiple strategy results for a specific metric (supports all four metrics)."""
    comparison_data = []
    
    for result in results_list:
        if result and 'metrics' in result:
            stats = result['metrics'][metric]
            comparison_data.append({
                'symphony': result['symphony'],
                'reliability': result['reliability'],
                f'iota_{metric}': stats['iota'],
                f'rating_{metric}': stats['persistence_rating'],
                f'oos_{metric}': result['oos_values'][metric],
                f'median_is_{metric}': stats['median_is'],
                'significant': stats['significant'],
                'p_value_adjusted': stats['p_value_adjusted'],
                'autocorr_adjustment': stats.get('autocorr_adjustment', 1.0),
                'bootstrap_method': stats.get('bootstrap_method', 'unknown')
            })
    
    return pd.DataFrame(comparison_data).sort_values(f'iota_{metric}', ascending=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — bye ✌️")
        
