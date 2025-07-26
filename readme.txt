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
