# Integrated Synthetic Population & Medicaid Policy Pipeline for Cancer Screening (Colon & Breast)

This repository contains a **fully configurable, CSV-driven pipeline** for generating synthetic populations and modeling the health policy impacts of Medicaid coverage changes on **colorectal cancer (CRC)** and **breast cancer** screening and treatment costs.

The pipeline now supports **two cancer types**:
- **Colorectal cancer (CRC)** — modeled via CCRAT-inspired parameters
- **Breast cancer** — modeled via BCRAT/Gail-inspired parameters

The pipeline consists of **4 integrated modules**:

1. **`model.py`** — Generates synthetic populations with demographics, screening status, and cancer risk (colon or breast)
2. **`screening_calculator.py`** — Encapsulates reusable screening logic for both `model.py` and `medicaid_policy_simulator.py` (cancer-agnostic)
3. **`medicaid_policy_simulator.py`** — Applies Medicaid policy scenarios and recomputes screening and economic costs (cancer-agnostic)
4. **`cancer_economics_model.py`** — Calculates healthcare costs based on screening status and cancer stage (cancer-agnostic)

---

## Overview

### Stage 1: Synthetic Population Generation (`model.py`)

Creates a geographically accurate synthetic population using **Iterative Proportional Fitting (IPF)** where:
- Aggregate counts by Age, Gender, and Race/Ethnicity match ACS marginals
- Joint distributions of Income, Education, and Health Insurance reflect Census microdata (ACS PUMS)

**Key inputs:**
- `data/demographics.csv` — ACS tract-level population counts
- `data/ipf-joint-distributions.csv` — Census-based conditional distributions (P(Income|Race), P(Education|Race), etc.)

The synthetic population is **shared** across colon and breast cancer simulations; only the downstream screening and risk/economic models differ by cancer type.

### Stage 2: Screening Status Assignment (`screening_calculator.py`, shared by `model.py` and `medicaid_policy_simulator.py`)

Assigns each individual a screening status (Screened/Not_Screened) that:
- Matches tract-level screening prevalence for the specified cancer type
- Reflects subgroup differences by age, race/ethnicity, and insurance (via BRFSS and related data)

**Key inputs (colon cancer):**
- `data/colon-rates.csv` — Tract-level baseline colonoscopy screening rates
- `data/colon-screening-joint-distributions.csv` — BRFSS-derived adjustment factors by age, race, insurance for CRC

**Key inputs (breast cancer):**
- `data/breast-cancer-rates.csv` — Tract-level baseline mammography screening rates
- `data/breast-screening-joint-distributions.csv` — BRFSS-derived adjustment factors by age, race, insurance for breast cancer

The **breast cancer screening data** (rates and joint adjustment factors) were constructed using the **same methodology and sources as the colon cancer data**, including:
- American Community Survey (ACS) tract-level demographics
- CDC PLACES tract-level preventive screening estimates
- Behavioral Risk Factor Surveillance System (BRFSS) patterns for age, race/ethnicity, and insurance

**Implementation:** The **`ScreeningCalculator`** class encapsulates all screening logic and is **cancer-agnostic**. Both `model.py` and `medicaid_policy_simulator.py` use this shared module, with behavior controlled by the `cancer_type` parameter (`colon` or `breast`).

### Stage 3: Cancer Risk Assessment (`model.py`)

Estimates each individual's 5-year unscreened and screened cancer risk using a **CCRAT-inspired model for colon** and a **BCRAT/Gail-inspired model for breast**.

**Key inputs (colon cancer):**
- `data/ccrat-parameters.csv` — Age-specific baseline CRC risks, multipliers for gender/race/income/education, screening effectiveness

**Key inputs (breast cancer):**
- `data/bcrat-parameters.csv` — Age-specific baseline breast cancer risks, multipliers for gender/race/income/education, screening effectiveness (Gail/BCRAT-style)

**Formula (conceptual, both cancers):**
```text
UnscreenedRisk = BaselineRisk(age) × Gender_Mult × Race_Mult × Income_Mult × Education_Mult
ScreenedRisk   = UnscreenedRisk × (1 - Screening_Effectiveness)
ScreeningBenefit = UnscreenedRisk - ScreenedRisk
```

The **risk model is parameterized entirely via CSV** and is agnostic to the cancer type beyond the specific parameter file (CCRAT vs BCRAT).

### Stage 4: Medicaid Policy Simulation & Economic Impact (`medicaid_policy_simulator.py` + `cancer_economics_model.py`)

**Applies policy scenarios that change Medicaid coverage**, then:
1. Recomputes screening status after coverage changes using `ScreeningCalculator` (for the selected cancer type)
2. Calculates expected healthcare costs using `CancerEconomicsModel`
3. Quantifies the **incremental treatment costs** due to missed screening

**Key inputs (colon cancer economics):**
- `data/colon_cancer_economics_parameters.csv` — Treatment costs by stage, screening cost (colonoscopy), stage distributions, discount rate, survival multipliers

**Key inputs (breast cancer economics):**
- `data/breast_cancer_economics_parameters.csv` — Treatment costs by stage (including Stage 0/DCIS), screening cost (mammography), stage distributions, discount rate, survival multipliers

As with screening, the **breast cancer economics parameters** were derived using the **same methodology and evidence base as the colon cancer parameters**, including SEER stage distributions, treatment cost literature, and standard health economics assumptions.

**Key capabilities:**
- CSV-driven policy definitions (no hard-coded scenarios)
- Medicaid inference from income and insurance status
- Screening recalculation after policy changes (colon or breast)
- **Economic impact analysis:** Shows both total cost changes AND the specific cost of missed screening

---

## Workflow

### Step 1: Run the Synthetic Population Pipeline (Colon or Breast)

#### Colon cancer example

```bash
python model.py   --cancer-type colon   --demographics data/demographics.csv   --ipf-joint-dist data/ipf-joint-distributions.csv   --screening-joint-dist data/colon-screening-joint-distributions.csv   --screening-rates data/colon-rates.csv   --risk-parameters data/ccrat-parameters.csv   --output output/synthetic_population_colon.csv   --scaling-factor 100
```

#### Breast cancer example

```bash
python model.py   --cancer-type breast   --demographics data/demographics.csv   --ipf-joint-dist data/ipf-joint-distributions.csv   --screening-joint-dist data/breast-screening-joint-distributions.csv   --screening-rates data/breast-cancer-rates.csv   --risk-parameters data/bcrat-parameters.csv   --output output/synthetic_population_breast.csv   --scaling-factor 100
```

**Output (colon example):** `output/synthetic_population_colon.csv` with columns:
- Demographic/SDOH: `Tract_GEOID`, `Age_Group`, `Race_Ethnicity`, `Income_Bracket`, `Education_Level`, `Health_Insurance_Status`
- Screening: `Age_Eligibility` (`Eligible_45-75` / `Outside_Eligible_45-75`), `Colon_Screening_Probability`, `Colon_Cancer_Screening_Status`
- CRC Risk: `Unscreened_Risk`, `Screened_Risk`, `Screening_Benefit`, `Risk_Category`

**Output (breast example):** `output/synthetic_population_breast.csv` with analogous columns, but:
- Screening: `Age_Eligibility` (`Eligible_40-74` / `Outside_Eligible_40-74`), `Breast_Screening_Probability`, `Breast_Cancer_Screening_Status`
- Breast cancer risk parameters from `bcrat-parameters.csv`

---

### Step 2: Run Medicaid Policy Simulation with Economic Analysis

#### Colon cancer example

```bash
python medicaid_policy_simulator.py   --cancer-type colon   --baseline-population output/synthetic_population_colon.csv   --screening-joint-dist data/colon-screening-joint-distributions.csv   --screening-rates data/colon-rates.csv   --economics-parameters data/colon_cancer_economics_parameters.csv   --policy-scenario income_tightening   --output-dir output/policy_colon_income
```

#### Breast cancer example

```bash
python medicaid_policy_simulator.py   --cancer-type breast   --baseline-population output/synthetic_population_breast.csv   --screening-joint-dist data/breast-screening-joint-distributions.csv   --screening-rates data/breast-cancer-rates.csv   --economics-parameters data/breast_cancer_economics_parameters.csv   --policy-scenario income_tightening   --output-dir output/policy_breast_income
```

**Outputs (per scenario):**
- `baseline_population_{cancer_type}.csv` — Baseline scenario with costs
- `policy_population_{cancer_type}_{scenario}.csv` — Policy scenario with updated coverage, screening, and costs
- `cost_comparison_{cancer_type}_{scenario}.csv` — Summary comparison between baseline and policy (coverage losses, treatment cost increase, net impact)

---

## Module Details

### `model.py` — Cancer-Agnostic Synthetic Population Generator

**Key Class:**
- `IntegratedSyntheticPopulationPipeline` — Generates synthetic population, assigns screening, and computes risk for either colon or breast cancer.

**Key Features:**
- `--cancer-type` controls whether colon or breast cancer parameters are used
- Uses `ScreeningCalculator` for screening assignment
- Uses CCRAT-style or BCRAT-style parameters depending on `--risk-parameters` file

**Inputs:**
- Common: `demographics.csv`, `ipf-joint-distributions.csv`
- Colon-specific: `colon-screening-joint-distributions.csv`, `colon-rates.csv`, `ccrat-parameters.csv`
- Breast-specific: `breast-screening-joint-distributions.csv`, `breast-cancer-rates.csv`, `bcrat-parameters.csv`

---

### `screening_calculator.py` — Shared, Cancer-Agnostic Screening Logic

**Purpose:** Encapsulates all screening status assignment logic so both `model.py` and `medicaid_policy_simulator.py` use identical methods across cancer types.

**Key Class:**
- `ScreeningCalculator` — Handles screening probability calculation and assignment.

**Key Method:**
- `assign_screening_to_population(df, insurance_column)` — Assigns screening status based on tract rates, age/race/insurance adjustments, and the specified insurance column; supports both colon and breast via `cancer_type` parameter.

**Usage (colon example):**
```python
from screening_calculator import ScreeningCalculator

calculator = ScreeningCalculator(
    cancer_type='colon',
    screening_joint_distributions_csv='data/colon-screening-joint-distributions.csv',
    screening_rates_csv='data/colon-rates.csv'
)

df_with_screening = calculator.assign_screening_to_population(
    df,
    insurance_column='Health_Insurance_Status'
)
```

**Usage (breast example):**
```python
breast_calc = ScreeningCalculator(
    cancer_type='breast',
    screening_joint_distributions_csv='data/breast-screening-joint-distributions.csv',
    screening_rates_csv='data/breast-cancer-rates.csv'
)

df_with_breast_screening = breast_calc.assign_screening_to_population(
    df,
    insurance_column='Health_Insurance_Status'
)
```

---

### `cancer_economics_model.py` — Cancer-Agnostic Healthcare Cost Calculator

**Purpose:** Calculates expected treatment and screening costs, and isolates the economic impact of missed screening, for both colon and breast cancer.

**Key Class:**
- `CancerEconomicsModel` — Loads parameters from CSV and computes costs for the specified `cancer_type`.

**Key Methods:**
- `calculate_individual_cost()` — Expected lifetime cost based on cancer risk, screening status, and stage distribution.
- `apply_costs_to_population()` — Applies cost calculation to an entire population dataframe.
- `generate_cost_report()` — Summary statistics (total, average, by screening status).
- `generate_scenario_comparison()` — Compares baseline vs policy, isolating treatment cost increases from missed screening.

**Parameters (CSV-driven):**
- Colon: `colon_cancer_economics_parameters.csv` (Stages I–IV, colonoscopy cost, stage distributions, discounting).
- Breast: `breast_cancer_economics_parameters.csv` (Stages 0–IV including DCIS, mammography cost, stage distributions, discounting).

The **breast cancer economics parameters** were derived using the **same literature-based approach** as colon (SEER, published cost studies, and standard discounting assumptions).

---

### `medicaid_policy_simulator.py` — Policy Simulator with Screening Recalculation (Colon & Breast)

**Purpose:** Applies Medicaid policy changes and recomputes screening + costs for **either colon or breast cancer**.

**Key Class:**
- `MedicaidPolicySimulator` — Manages policy application, screening reassignment, and economic analysis.

**Key Methods:**
- `simulate_income_tightening()` — Tightens income eligibility (e.g., from 138% to 100% FPL).
- `simulate_asset_test()` — Adds an asset test (proxy implemented via higher income brackets).
- `simulate_work_requirement()` — Adds a work requirement (modeled as 15% of Medicaid population losing coverage).
- `reassign_screening()` — Uses `ScreeningCalculator` to recompute screening after coverage changes.
- `calculate_costs()` — Uses `CancerEconomicsModel` to compute baseline vs policy costs.
- `run_scenario()` — End-to-end scenario driver (policy + screening + costs + outputs).

**Inputs:**
- Baseline population from `model.py` for the chosen cancer type.
- Cancer-specific screening joint distributions and rates.
- Cancer-specific economics parameters.

---

## CSV Configuration Files

### Core Input Data Files

| File | Purpose | Notes |
|------|---------|-------|
| `data/demographics.csv` | ACS tract-level population marginals | Shared across cancer types |
| `data/ipf-joint-distributions.csv` | Census-based conditional distributions | Shared across cancer types |
| `data/colon-rates.csv` | Tract-level baseline colonoscopy screening rates | Colon-specific |
| `data/colon-screening-joint-distributions.csv` | CRC screening adjustment factors by age/race/insurance | Colon-specific, BRFSS/PLACES-based |
| `data/breast-cancer-rates.csv` | Tract-level baseline mammography screening rates | Breast-specific, constructed using same ACS/CDC/BRFSS methodology as colon |
| `data/breast-screening-joint-distributions.csv` | Breast screening adjustment factors by age/race/insurance | Breast-specific, constructed using same ACS/CDC/BRFSS methodology as colon |
| `data/ccrat-parameters.csv` | CCRAT-style CRC risk parameters | Colon-specific |
| `data/bcrat-parameters.csv` | BCRAT/Gail-style breast cancer risk parameters | Breast-specific |
| `data/colon_cancer_economics_parameters.csv` | CRC treatment and screening economics | Colon-specific |
| `data/breast_cancer_economics_parameters.csv` | Breast cancer treatment and screening economics | Breast-specific |

---

## Data Sources & Evidence Base

The **breast cancer data and parameters were developed using the same methodological framework and primary data sources as the colon cancer components**, namely:

- **American Community Survey (ACS)** for tract-level demographics and social determinants.
- **ACS Public Use Microdata Sample (PUMS)** for conditional distributions (P(Income|Race), P(Education|Race), P(Insurance|Race,Age)).
- **CDC PLACES** for tract-level preventive screening estimates (colonoscopy and mammography).
- **BRFSS** for age, race/ethnicity, and insurance gradients in screening behavior (both colon and breast screening).
- **SEER** and **NCI CCRAT/BCRAT documentation** for baseline risks, stage distributions, and incidence patterns.
- **Health economics literature** for stage-specific treatment costs, screening costs, and discounting assumptions.

These sources are used **consistently across both cancers** so that colon and breast simulations are methodologically comparable.

---

## Key Design Principles

- **Fully CSV-Driven:** All behavioral, screening, risk, and cost parameters live in CSV files. No hard-coded constants are required to switch between colon and breast cancer.
- **Cancer-Agnostic Core:** The main Python modules (`model.py`, `screening_calculator.py`, `cancer_economics_model.py`, `medicaid_policy_simulator.py`) are cancer-agnostic and controlled entirely by the `--cancer-type` option and the CSV file choices.
- **Evidence-Based:** Both colon and breast cancer models are grounded in ACS, CDC PLACES, BRFSS, SEER, and peer-reviewed literature.
- **Transparent Economic Interpretation:** Explicit separation of **treatment cost increases** (from missed screening) vs. **apparent savings** (from avoiding screening procedures), for both cancers.

---

## Questions & Customization

To customize or extend the pipeline:
1. Modify the CSV configuration files for new geographies, updated screening rates, or alternative risk/economic assumptions.
2. Add new policy scenarios by extending `MedicaidPolicySimulator` or integrating CSV-driven policy definitions.
3. Add additional cancer types by supplying new screening rate/joint-distribution CSVs, risk parameter CSVs, and economics parameter CSVs, and then extending the `cancer_type` handling where appropriate.

This design is intended to make it straightforward to maintain methodological consistency while expanding to new cancers, geographies, or policy scenarios.
