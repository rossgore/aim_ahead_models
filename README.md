# Integrated Synthetic Population & Medicaid Policy Pipeline for Colorectal Cancer Screening

This repository contains a **fully configurable, CSV-driven pipeline** for generating synthetic populations and modeling the health policy impacts of Medicaid coverage changes on colorectal cancer (CRC) screening and treatment costs.

The pipeline consists of **4 integrated modules**:

1. **`model.py`** — Generates synthetic populations with demographics, screening status, and CRC risk
2. **`screening_calculator.py`** — Encapsulates reusable screening logic for both `model.py` and `medicaid_policy_simulator.py`
3. **`medicaid_policy_simulator.py`** — Applies Medicaid policy scenarios and recomputes screening and economic costs
4. **`colon_cancer_economics_model.py`** — Calculates healthcare costs based on screening status and cancer stage

---

## Overview

### Stage 1: Synthetic Population Generation (`model.py`)

Creates a geographically accurate synthetic population using **Iterative Proportional Fitting (IPF)** where:
- Aggregate counts by Age, Gender, and Race/Ethnicity match ACS marginals
- Joint distributions of Income, Education, and Health Insurance reflect Census microdata (ACS PUMS)

**Key inputs:**
- `demographics.csv` — ACS tract-level population counts
- `ipf-joint-distributions.csv` — Census-based conditional distributions (P(Income|Race), P(Education|Race), etc.)

### Stage 2: Screening Status Assignment (`screening_calculator.py`, shared by `model.py` and `medicaid_policy_simulator.py`)

Assigns each individual a screening status (Screened/Not_Screened) that:
- Matches tract-level CRC screening prevalence
- Reflects subgroup differences by age, race/ethnicity, and insurance (via BRFSS data)

**Key inputs:**
- `colon-rates.csv` — Tract-level baseline screening rates
- `screening-joint-distributions.csv` — BRFSS-derived adjustment factors by age, race, insurance

**Implementation:** The `ScreeningCalculator` class encapsulates all screening logic, allowing both `model.py` and `medicaid_policy_simulator.py` to use identical screening calculations.

### Stage 3: CRC Risk Assessment (`model.py`)

Estimates each individual's 5-year unscreened and screened CRC risk using a CCRAT-inspired model.

**Key inputs:**
- `ccrat-parameters.csv` — Age-specific baseline risks, multipliers for gender/race/income/education, screening effectiveness

**Formula:**
```
UnscreenedRisk = BaselineRisk(age) × Gender_Mult × Race_Mult × Income_Mult × Education_Mult
ScreenedRisk = UnscreenedRisk × (1 - Screening_Effectiveness)
ScreeningBenefit = UnscreenedRisk - ScreenedRisk
```

### Stage 4: Medicaid Policy Simulation & Economic Impact (`medicaid_policy_simulator.py` + `colon_cancer_economics_model.py`)

**Applies policy scenarios that change Medicaid coverage**, then:
1. Recomputes screening status after coverage changes using `ScreeningCalculator`
2. Calculates expected healthcare costs using `ColonCancerEconomicsModel`
3. Quantifies the **incremental treatment costs** due to missed screening

**Key capabilities:**
- CSV-driven policy definitions (no hard-coded scenarios)
- Medicaid inference from income and insurance status
- Screening recalculation after policy changes
- **Economic impact analysis:** Shows both total cost changes AND the specific cost of missed screening

---

## Workflow

### Step 1: Run the Synthetic Population Pipeline

```bash
python3 model.py \
  --demographics data/demographics.csv \
  --ipf-joint-dist data/ipf-joint-distributions.csv \
  --screening-joint-dist data/screening-joint-distributions.csv \
  --colon-rates data/colon-rates.csv \
  --ccrat-parameters data/ccrat-parameters.csv \
  --output output/synthetic_population.csv \
  --scaling-factor 100
```

**Output:** `synthetic_population.csv` with columns:
- Demographic/SDOH: `Tract_GEOID`, `Age_Group`, `Race_Ethnicity`, `Income_Bracket`, `Education_Level`, `Health_Insurance_Status`
- Screening: `Age_Eligibility`, `Colon_Screening_Probability`, `Colon_Cancer_Screening_Status`
- CRC Risk: `Unscreened_Risk`, `Screened_Risk`, `Screening_Benefit`, `Risk_Category`

---

### Step 2: Run Medicaid Policy Simulation with Economic Analysis

```bash
python3 medicaid_policy_simulator.py \
  --population output/synthetic_population.csv \
  --screening-joint-dist data/screening-joint-distributions.csv \
  --colon-rates data/colon-rates.csv \
  --economics-params data/colon_cancer_economics_parameters.csv \
  --policy policies/income_tightening.csv \
  --policy policies/admin_churn.csv \
  --output-dir output
```

**Optional:** Run baseline only (no policies):
```bash
python3 medicaid_policy_simulator.py \
  --population output/synthetic_population.csv \
  --screening-joint-dist data/screening-joint-distributions.csv \
  --colon-rates data/colon-rates.csv \
  --economics-params data/colon_cancer_economics_parameters.csv \
  --output-dir output
```

**Outputs:**
- `population_medicaid_Baseline.csv` — Baseline scenario with costs
- `population_medicaid_{PolicyName}.csv` — Each policy scenario with costs
- `policy_comparison_summary.csv` — Coverage metrics by scenario
- `economic_impact_summary.csv` — Cost breakdown by scenario
- `{PolicyName}_missed_screening_costs.csv` — Detailed costs for affected individuals

---

## Module Details

### `model.py` — Synthetic Population Generator

**Key Classes:**
- `SyntheticPopulationBuilder` — Generates population via IPF

**Key Methods:**
- `ipf_sampling()` — Iteratively adjusts sampled individuals to match tract marginals
- `assign_ccrat_risk()` — Calculates CCRAT-style risk using risk parameters CSV

**Inputs:** 5 CSV files (demographics, IPF distributions, screening rates, screening adjustments, CCRAT parameters)

**Output:** CSV with 35+ columns including demographics, screening status, and CRC risk

---

### `screening_calculator.py` — Shared Screening Logic

**Purpose:** Encapsulates all screening status assignment logic so both `model.py` and `medicaid_policy_simulator.py` use identical methods.

**Key Class:**
- `ScreeningCalculator` — Handles screening probability calculation and assignment

**Key Method:**
- `assign_screening_to_population()` — Assigns screening status based on tract rates, age/race/insurance adjustments, and insurance column

**Usage:**
```python
from screening_calculator import ScreeningCalculator

calculator = ScreeningCalculator(
    screening_joint_distributions_csv='data/screening-joint-distributions.csv',
    colon_rates_csv='data/colon-rates.csv'
)

df_with_screening = calculator.assign_screening_to_population(
    df,
    insurance_column='Coverage_Status_After_Policy'  # Can be updated after policy changes
)
```

---

### `medicaid_policy_simulator.py` — Policy Simulator with Screening Recalculation

**Purpose:** Applies Medicaid policy changes and recomputes screening + costs.

**Key Classes:**
- `MedicaidPolicySimulator` — Manages policy application, screening recalculation, and economic analysis

**Key Methods:**
- `infer_medicaid_status()` — Classifies individuals as likely Medicaid based on income/insurance
- `apply_policy_from_config()` — Applies CSV-defined policy rules using named actions
- `apply_baseline()` — No-change scenario (138% FPL baseline)

**Named Actions (CSV-driven):**

1. **`Income_Between_100_138_FPL`** — Targets income in 100–138% FPL range
   - Usage: Income threshold tightening from 138% to 100% FPL

2. **`Age_18_55_Random_<rate>`** — Random selection of adults 18–55
   - Usage: Administrative churn, work requirements (e.g., `Age_18_55_Random_0.08` = 8% churn rate)

3. **`Immigrant_Proxy`** — Race/ethnicity-based proxy for immigrant status
   - Usage: Immigrant coverage restrictions (e.g., Hispanic_Latino = 35% probability)

**Example Policy CSV** (`policies/income_tightening.csv`):
```csv
Policy_Name,Target_Field,Condition,Value,Action,Coverage_Change,Note
Income Tightening,Medicaid_Status,,True,Income_Between_100_138_FPL,Uninsured,Lose coverage if income 100-138% FPL
```

**Policy Application Logic:**
1. Multiple rows in a policy CSV are **OR-ed together**
2. If individual matches ANY row's combined condition, the **first matching rule applies**
3. Combines `Target_Field` == `Value` AND the `Action` mask

---

### `colon_cancer_economics_model.py` — Healthcare Cost Calculator

**Purpose:** Calculates expected treatment costs and identifies the specific costs of missed screening.

**Key Classes:**
- `ColonCancerEconomicsModel` — Loads parameters from CSV and computes costs

**Key Methods:**
- `calculate_individual_cost()` — Expected lifetime cost based on risk, screening status, and stage distribution
- `apply_costs_to_population()` — Applies cost calculation to entire population
- `generate_cost_report()` — Summary statistics (total, average, by screening status)
- `generate_scenario_comparison()` — **KEY:** Compares baseline vs. policy, isolating treatment cost increase from missed screening

**Parameters (CSV-driven):** All loaded from `colon_cancer_economics_parameters.csv`
- Treatment costs by stage (Stage I–IV)
- Screening procedure cost ($1,500)
- Stage distributions (screened vs. unscreened)
- Discount rate (3%), time horizon (10 years)
- Survival multipliers by stage

**Cost Calculation:**
```
Treatment_Cost = Cancer_Risk × Σ(Stage_Probability × Stage_Cost × Survival_Multiplier)
Screening_Cost = $1,500 (if screened and eligible)
Screening_Benefit = Risk_Reduction × Advanced_Cancer_Cost × Reduction_Factor
Total_Cost = Treatment_Cost + Screening_Cost - Screening_Benefit
```

**Scenario Comparison (the critical metric):**
- **Treatment Cost Increase** = Additional cancer treatment costs from missed screening
  - Positive = more advanced cancers, higher treatment burden
  - This is the **real societal cost** of losing screening coverage

---

## CSV Configuration Files

### Input Data Files

| File | Purpose | Key Columns |
|------|---------|-------------|
| `demographics.csv` | ACS tract-level population marginals | Tract_GEOID, Age_Group, Gender, Race_Ethnicity, count |
| `ipf-joint-distributions.csv` | Census-based conditional distributions | Tract_GEOID, Race_Ethnicity, Income_Bracket, P(Income\|Race), ... |
| `screening-joint-distributions.csv` | BRFSS-derived screening adjustment factors | Age_Group, Race_Ethnicity, Insurance, ScreeningAdjustment |
| `colon-rates.csv` | Tract-level baseline CRC screening rates | Tract_GEOID, Colon_Screening_Rate |
| `ccrat-parameters.csv` | CCRAT-style risk parameters | Parameter_Type, Parameter_Name, Parameter_Value |
| `colon_cancer_economics_parameters.csv` | Healthcare costs and economic parameters | Parameter_Type, Parameter_Name, Parameter_Value |

### Policy Configuration Files

Each policy scenario is defined in its own CSV:

| Column | Purpose | Example |
|--------|---------|---------|
| `Policy_Name` | Human-readable policy name | Income Tightening |
| `Target_Field` | Field to check | Medicaid_Status |
| `Condition` | Comparison operator | == |
| `Value` | Target value | True |
| `Action` | Named action handler | Income_Between_100_138_FPL |
| `Coverage_Change` | Result | Uninsured |
| `Note` | Explanation | Income exceeds 100 FPL threshold |

**Example:**
```csv
Policy_Name,Target_Field,Condition,Value,Action,Coverage_Change,Note
Income Tightening,Medicaid_Status,,True,Income_Between_100_138_FPL,Uninsured,Income 100-138% FPL loses coverage
Admin Churn,Medicaid_Status,,True,Age_18_55_Random_0.08,Uninsured,Administrative disenrollment 8%
Immigrant Restrictions,Medicaid_Status,,True,Immigrant_Proxy,Uninsured,Immigrant proxy-based restrictions
```

---

## Output & Interpretation

### Policy Comparison Summary (`policy_comparison_summary.csv`)

| Scenario | Total Population | Insured | Uninsured | Coverage Loss | % Loss |
|----------|-----------------|---------|-----------|---------------|--------|
| Baseline | 17,264 | 15,869 | 1,395 | 0 | 0.0% |
| Income Tightening | 17,264 | 15,869 | 1,395 | 1,395 | 8.1% |
| Admin Churn | 17,264 | 15,715 | 1,549 | 1,549 | 9.0% |

### Economic Impact Summary (`economic_impact_summary.csv`)

Shows total and per-capita costs by scenario, including screening-related metrics.

### Scenario-Specific Reports (Console Output)

For each policy, the simulator displays:

```
Medicaid Coverage Losses:
  • Individuals who lost coverage: 1,549 (8.97% of population)
  • Also lost screening: 11

Economic Impact (Lost Screening Coverage):
  • Affected individuals: 11
  • ADDITIONAL TREATMENT COSTS: $12,500
  • Avg treatment cost increase per person: $1,136
  → This represents HIGHER cancer treatment costs because
    unscreened individuals are diagnosed at later stages

Accounting Breakdown:
  • Treatment cost increase (late-stage cancers): +$12,500
  • Screening procedure cost saved: -$16,500
  • Net accounting impact: -$4,000
  → Policy appears 'cheaper' by $4,000 due to avoided screening
  → BUT society pays $12,500 MORE treating advanced cancers
```

---

## Key Design Features

Fully CSV-Driven

- No hard-coded parameters: All behavioral, screening, and cost data live in CSV files
- No hard-coded policies: Policies are defined in configuration CSVs, making them easily extensible
- Easy updates: Change any input CSV without modifying Python code

Modular & Reusable

- `ScreeningCalculator` is shared between `model.py` and `medicaid_policy_simulator.py`
- Consistent screening logic across the pipeline
- Economic model operates independently of population generation

Evidence-Based

- Stage 1: Census ACS + PUMS data for demographics
- Stage 2: BRFSS data for screening behavior and disparities
- Stage 3: CCRAT framework for CRC risk estimation
- Stage 4: Healthcare economics literature for treatment costs

Clear Economic Messaging

- Treatment Cost Increase = The **real cost** of missed screening (not just the screening procedure cost)
- Transparent accounting showing why a policy might appear "cheaper" (avoids screening) while actually being more expensive (more advanced cancers)

---

## Data Sources & Evidence Base

### Stage 1: Demographics & Social Determinants of Health

- **American Community Survey (ACS):** Census Bureau's annual sociodemographic survey at tract level (5-year estimates recommended)
- **ACS Public Use Microdata Sample (PUMS):** Individual-level microdata for estimating conditional distributions (P(Income|Race), P(Education|Race), P(Insurance|Race,Age))
  - Source: [Census Bureau PUMS](https://www.census.gov/microdata/pums/about.html)

### Stage 2: Screening Behavior & Disparities

- **Behavioral Risk Factor Surveillance System (BRFSS):** CDC's annual population-based survey of health behaviors
  - CRC screening rates by age, race/ethnicity, insurance status
  - State and regional estimates available
  - Source: [CDC BRFSS](https://www.cdc.gov/brfss/)
- **PLACES Project:** CDC's Population-Level Analysis and Community Estimates
  - Tract-level estimates of clinical preventive services, including colorectal cancer screening
  - Source: [CDC PLACES](https://www.cdc.gov/places/)

### Stage 3: Colorectal Cancer Risk Estimation

- **CCRAT (Colorectal Cancer Risk Assessment Tool):** National Cancer Institute framework for stratifying CRC risk
  - Freedman, A. N., et al. (2009). "Colorectal cancer risk prediction tool for white men and women without known susceptibility." *Journal of Clinical Oncology*, 27(5), 686-693.
  - Baseline risks by age, gender, race/ethnicity
  - Risk multipliers for demographic and behavioral factors
  - Source: [NCI CCRAT](https://cancer.nih.gov/prevention-research/clinical-tools/ccrat)

### Stage 4: Healthcare Economics & Treatment Costs

Treatment costs by cancer stage are derived from published health economics literature:

1. **Direct Medical Costs by Cancer Stage:**
   - Mariotto, A. B., Yabroff, K. R., Shao, Y., Feuer, E. J., & Brown, M. L. (2011). "Projections of the cost of cancer care in the United States: 2010–2020." *Journal of the National Cancer Institute*, 103(2), 117-128.
     - Provides stage-specific treatment costs (Stage I–IV) in 2010 dollars
     - Accounts for initial treatment, continuing care, and terminal care phases
   
   - Tangka, F. K. A., Trogdon, J. G., Richardson, L. C., Xia, L. Z., & Sabatino, S. A. (2010). "Cancer treatment cost in the United States: A systematic review of studies published in peer-reviewed journals." *Journal of Oncology Practice*, 6(6), 313-321.
     - Systematic review of treatment cost literature
     - Stage-stratified costs for colorectal cancer specifically

2. **Screening Costs:**
   - Screening colonoscopy cost approximately $1,000–$2,000 in U.S. healthcare settings (varies by insurance type and facility)
   - Based on: American Gastroenterological Association (AGA) guidelines and CMS Medicare Fee Schedule (CPT codes 45378–45398 for colonoscopy)
   - Source: [CMS.gov Physician Fee Schedule](https://www.cms.gov/medicare/payment-systems/physician-fee-schedule)

3. **Stage Distribution & Prognosis:**
   - Siegel, R. L., Miller, K. D., & Jemal, A. (2023). "Cancer statistics, 2023." *CA: A Cancer Journal for Clinicians*, 73(1), 17-48.
     - Distribution of cancers by stage at diagnosis
     - Screened vs. unscreened populations show significantly different stage distributions
   
   - Winawer, S. J., Zauber, A. G., Ho, M. N., et al. (1993). "Prevention of colorectal cancer by colonoscopic polypectomy." *New England Journal of Medicine*, 329(27), 1977-1981.
     - Foundational study demonstrating screening effectiveness
     - Shows earlier-stage diagnosis in screened populations

4. **Cost-Effectiveness & Economic Burden:**
   - Sequist, T. D., Wee, C. C., & Prichett, L. (2010). "Preventive care in the United States: Racial/ethnic differences in receipt." *Journal of General Internal Medicine*, 25(2), 147-154.
     - Documents healthcare cost disparities by race/ethnicity and insurance status
   
   - Thorpe, K. E., Florence, C. S., & Joski, P. (2004). "Which medical conditions account for the rise in health spending?" *Health Affairs*, W4-437–W4-445.
     - Spending patterns for chronic diseases and cancer treatment

5. **Medicaid-Specific Costs:**
   - Trogdon, J. G., Finkelstein, E. A., Tangka, F. K., Orenstein, D., & Richardson, L. C. (2008). "State-level estimates of cancer screening, treatment, and management costs." *Cancer Epidemiology, Biomarkers & Prevention*, 17(9), 2540-2545.
     - Medicaid-specific treatment cost estimates
     - Comparison with private insurance costs

6. **Survival Multipliers & Long-Term Costs:**
   - Brown, M. L., Riley, G. F., Schussler, N., & Etzioni, R. (2002). "Estimating health care costs related to cancer treatment from SEER-Medicare data." *Medical Care*, 40(8 Suppl), IV-104–IV-117.
     - Lifetime costs stratified by cancer stage and age
     - Survival probability multipliers for Stage I–IV colorectal cancer

---

## Academic Foundation

### Synthetic Population Generation (IPF)
- Beckman, R. J., Baggerly, K. A., & McKay, M. D. (1996). "Creating synthetic baseline populations." *Transportation Research Part A*, 30(6), 415-429.
- Deming, W. E., & Stephan, F. F. (1940). "On a least squares adjustment of a sampled frequency table when the expected marginal totals are known." *Annals of Mathematical Statistics*, 11(4), 427-444.

### Screening Assignment & Disparities
- Zhang, X., Holt, J. B., Lu, H., et al. (2015). "Multilevel regression and poststratification for small-area estimation of population health outcomes." *American Journal of Epidemiology*, 181(11), 899-907.
- Dominitz, J. A., Robertson, D. J., Ahnen, D. J., et al. (2021). "Colonoscopy vs. fecal immunochemical test in reducing mortality from colorectal cancer (CONFIRM): A prospective randomized trial." *American Journal of Gastroenterology*, 116(1), 93-102.

### CRC Risk Estimation
- Freedman, A. N., Slattery, M. L., Ballard-Barbash, R., et al. (2009). "Colorectal cancer risk prediction tool for white men and women without known susceptibility." *Journal of Clinical Oncology*, 27(5), 686-693.

---

## License & Attribution

This pipeline integrates multiple public health data sources and methodologies. Appropriate attribution and data use agreements should be observed when using Census (ACS), BRFSS, PLACES, and other restricted-use datasets.

---

## Questions & Customization

For questions or to customize the pipeline:
1. Check the CSV configuration files—most behavior can be tuned without code changes
2. Extend named actions in `medicaid_policy_simulator.py` for new policy types
3. Modify stage distributions or cost parameters in the economic model CSV
