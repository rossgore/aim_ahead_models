# Integrated Synthetic Population & CRC Risk Pipeline

## Overview

This repository contains `model.py`, a Python pipeline that generates geographically accurate synthetic populations for public health modeling. It is currently configured for **colorectal cancer (CRC)** screening and risk assessment at the **census tract** level.

The pipeline has three independent, data-driven stages:

1. Synthetic population generation using demographic data.
2. Screening status assignment based on regional screening behavior.
3. CRC risk estimation using a CCRAT-inspired risk model.

All behavioral and risk parameters are configured via external CSV files, so they can be updated without modifying the source code.

***

## High-Level Pipeline

The model executes in three stages:

1. **Stage 1 – IPF-based synthetic population generation**  
   Uses Census/ACS demographics and **IPF joint distributions** estimated from Census-based microdata (e.g., ACS PUMS) for Virginia or the southeast Virginia region.

2. **Stage 2 – Screening status assignment**  
   Uses tract-level screening rates and **screening joint adjustment factors** derived from **BRFSS** (Behavioral Risk Factor Surveillance System) for adults in the **southeast region of Virginia** (Richmond + Hampton Roads).

3. **Stage 3 – CCRAT risk assessment**  
   Uses a CSV of CCRAT-style parameters (age-specific baseline risks and multipliers) to estimate unscreened and screened 5‑year CRC risk for each individual.

***

## Stage 1: Synthetic Population Generation (IPF)

### Goal

Create a synthetic population of individuals (agents) such that:

- Aggregate counts by **Age**, **Gender**, and **Race/Ethnicity** match the ACS marginals for each tract.
- Joint distributions of **Income**, **Education**, and **Health Insurance** are realistic and consistent with those observed in Census-based microdata.

### Method

The model uses **Iterative Proportional Fitting (IPF)** and sampling to construct individuals who jointly satisfy:

- Tract-level ACS marginals from `demographics.csv`.
- Cross-variable relationships embedded in `ipf-joint-distributions.csv`.

### Data Sources and Construction of `ipf-joint-distributions.csv`

The file `ipf-joint-distributions.csv` supplies conditional distributions such as:

- \(P(\text{Income} \mid \text{Race})\)
- \(P(\text{Education} \mid \text{Race})\)
- \(P(\text{Insurance} \mid \text{Race}, \text{AgeGroup})\)

These joint distributions should be constructed from **Census-based data**, not BRFSS:

- **Recommended primary source:**  
  **ACS Public Use Microdata Sample (PUMS)** for Virginia, or restricted to PUMAs that approximate the **southeast Virginia region** (Richmond + Hampton Roads). From PUMS, one can directly estimate:
  - Empirical \(P(\text{Income} \mid \text{Race})\) by counting respondents by race and income bracket.
  - Empirical \(P(\text{Education} \mid \text{Race})\).
  - Empirical \(P(\text{Health Insurance} \mid \text{Race}, \text{AgeGroup})\) (or at least by age and insurance).

- **Aligning with tract-level ACS:**
  - The **marginals** (e.g., total by race, total by age, total by income) come from tract-level ACS summary tables in `demographics.csv`.
  - The **conditional relationships** (how income, education, and insurance relate to race and age) are taken from PUMS at the PUMA/region level and written into `ipf-joint-distributions.csv`.
  - The IPF procedure then iteratively adjusts sampled individuals to match the tract marginals while preserving these census-derived relationships as much as possible.

In other words:

- **`demographics.csv`** ⇒ “How many people of each type live in each tract?”
- **`ipf-joint-distributions.csv` (from ACS PUMS)** ⇒ “Given race and age, how likely is each income, education, and insurance status in this region?”

This keeps **structural sociodemographic relationships grounded in Census data**, which is appropriate for Stage 1.

### Implementation Notes

- The code defines lists for:
  - `self.age_groups`
  - `self.races`
  - `self.income_brackets`
  - `self.education_levels`
- For each tract, the model:
  1. Computes age×gender and race marginals from `demographics.csv`.
  2. Reads conditional distributions from that tract’s row in `ipf-joint-distributions.csv`.
  3. Samples individual agents via an IPF-inspired procedure until the synthetic distribution matches the marginals within a tolerance.

**Academic foundation:**

- Beckman, R. J., Baggerly, K. A., & McKay, M. D. (1996). *Creating synthetic baseline populations*. Transportation Research Part A.
- Deming, W. E., & Stephan, F. F. (1940). *On a least squares adjustment of a sampled frequency table when the expected marginal totals are known*. Annals of Mathematical Statistics.

***

## Stage 2: Screening Status Assignment (BRFSS-Based)

### Goal

Assign each individual a colon cancer screening status (`Screened` or `Not_Screened`) in a way that:

- Matches tract-level CRC screening prevalence from `colon-rates.csv`.
- Reflects subgroup differences in screening uptake by Age, Race/Ethnicity, and Insurance, as observed in **BRFSS** for the **southeast Virginia region**.

### Data Inputs

- **`colon-rates.csv`**  
  Contains:
  - `GEOID`
  - `COLON_SCREEN_RATE` (e.g., percentage up-to-date with CRC screening)
- **`screening-joint-distributions.csv`**  
  Contains adjustment columns such as:
  - `45to49_Screening_Adjustment`, `50to54_Screening_Adjustment`, …
  - `Insured_Screening_Adjustment`, `Uninsured_Screening_Adjustment`
  - `White_NonHispanic_Screening_Adjustment`, `Black_NonHispanic_Screening_Adjustment`, etc.

### Source of Screening Joint Distributions

Here, **BRFSS is the right source**:

- Use BRFSS microdata for **adult residents in southeast Virginia** (Richmond + Hampton Roads).
- From BRFSS, estimate **relative** screening propensities:
  - By Age group (45–49, 50–54, …).
  - By Race/Ethnicity.
  - By Insurance status.

These relative effects become multiplicative **adjustment factors** in `screening-joint-distributions.csv`. For example:

- If BRFSS shows that:
  - Insured individuals have 1.3× the odds of being screened compared to uninsured.
  - Adults 65–75 have 1.5× the odds compared to 45–49-year-olds.
  - A particular race group has 0.8× the odds compared to White non-Hispanic.
- Then these factors can be encoded as:
  - `Insured_Screening_Adjustment = 1.3`
  - `Uninsured_Screening_Adjustment = 1.0` (reference)
  - `65to69_Screening_Adjustment = 1.5`
  - `45to49_Screening_Adjustment = 1.0` (reference)
  - etc.

### Implementation

For each individual:

1. Determine tract-level baseline \(p_{\text{tract}}\) from `colon-rates.csv`.
2. Look up age, race, and insurance adjustment factors from `screening-joint-distributions.csv`.
3. Multiply:
   \[
   p_{\text{indiv}} = \text{clip}_{[0.01,0.99]}\left(p_{\text{tract}} \times \text{AgeAdj} \times \text{RaceAdj} \times \text{InsuranceAdj}\right)
   \]
4. Draw a Bernoulli random variable with probability \(p_{\text{indiv}}\) to assign `Screened` / `Not_Screened`.

This cleanly separates:

- **Level 2 (tract)** information from PLACES/other sources.
- **Level 1 (individual)** relative effects estimated from **BRFSS regional survey data**.

**Academic foundation:**

- Zhang, X., Holt, J. B., et al. (2014). *Multilevel regression and poststratification for small-area estimation of population health outcomes*. American Journal of Epidemiology.
- Numerous CRC screening disparity studies using BRFSS data to quantify differences by race, income, education, and insurance.

***

## Stage 3: CCRAT Risk Assessment (CSV-Driven)

### Goal

Estimate each individual’s 5‑year CRC risk under:

- No screening (`Unscreened_Risk`)
- Given screening effectiveness (`Screened_Risk`)

and quantify `Screening_Benefit`.

### Data Source

- **`ccrat-parameters.csv`**  
  Structured with:
  - `Parameter_Category`: e.g., `age_baseline_risk`, `gender_multiplier`, `race_multiplier`, `income_multiplier`, `education_multiplier`, `screening_effectiveness`.
  - `Parameter_Name`: e.g., `risk_45to49`, `male`, `white`, `under20k`, `bachelorplus`.
  - `Parameter_Value`: numeric.

### Implementation

- Age, gender, race, income, and education are mapped to CCRAT-compatible categories.
- Risk is computed as:
  - \( \text{Unscreened\_Risk} = \text{BaselineRisk(age)} \times \text{GenderMult} \times \text{RaceMult} \times \text{IncomeMult} \times \text{EducationMult} \)
  - \( \text{Screened\_Risk} = \text{Unscreened\_Risk} \times (1 - \text{ScreeningEffectiveness}) \)
  - `Screening_Benefit` = difference between the two.
- Risk category is assigned based on thresholds.

**Academic foundation:**

- Freedman, A. N., et al. (2009). *Colorectal cancer risk prediction tool for white men and women without known susceptibility*. Journal of Clinical Oncology.
- General literature on CRC risk models and absolute risk estimation.

***

## Input Arguments and Execution

Run the model as:

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

- `--demographics`: ACS tract-level marginals (Census-based).
- `--ipf-joint-dist`: Joint distributions estimated from ACS PUMS or equivalent Census microdata for Virginia/southeast Virginia.
- `--screening-joint-dist`: BRFSS-derived relative screening adjustment factors for the southeast Virginia region.
- `--colon-rates`: Tract-level baseline CRC screening rates.
- `--ccrat-parameters`: CSV configuration for CCRAT-style risk parameters.
- `--scaling-factor`: Controls the synthetic sample size relative to real population.

***

## Output

The model writes a CSV of synthetic individuals with:

- Tract identifiers and names.
- Demographic and SDOH attributes consistent with ACS marginals and Census-based joint distributions.
- Screening status probabilities and realized screening outcomes, consistent with tract rates and BRFSS-derived disparities.
- CCRAT-style unscreened and screened risk, plus screening benefit and risk category.

This separation of data sources:

- **Census/ACS/PUMS → `demographics.csv` and `ipf-joint-distributions.csv`** for structural sociodemographics.
- **BRFSS → `screening-joint-distributions.csv`** for screening behavior patterns.
- **Risk literature → `ccrat-parameters.csv`** for disease risk modeling.

provides a coherent, evidence-based foundation for the integrated pipeline.