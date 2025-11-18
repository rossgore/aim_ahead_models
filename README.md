# CCRAT-Enhanced Census Tract Model

## Overview

The CCRAT-Enhanced Census Tract Model is an open-source, modular Python package for simulating the effect of health policy changes (such as Medicaid disenrollment/unwinding) on colorectal cancer risk, screening, and costs. The model achieves high spatial resolution by operating at the census tract level and allows for detailed demographic stratification using the National Cancer Institute (NCI) Colorectal Cancer Risk Assessment Tool (CCRAT) methodology. The approach is ideal for simulation-based decision support for health interventions, policy evaluation, and geographically targeted planning.

---

## Modeling Strategy

### **Component 1: Demographic Microsimulation**
- **Purpose:**  
  To generate synthetic but statistically representative populations for each census tract matching real demographic distributions.
- **Inputs:**  
  Census tract demographic tables such as population by age, race/ethnicity, gender, income, and education.
- **Outputs:**  
  A dataframe of individual synthetic persons per tract with associated demographic attributes.

### **Component 2: Risk Assessment (CCRAT)**
- **Purpose:**  
  For every synthetic individual, calculate their 5-year risk of developing colorectal cancer based on age, gender, race, income, and education, using published CCRAT formulae and multipliers.
- **Inputs:**  
  Synthetic population microdata.  
  Parameter JSON encoding CCRAT baseline risks and multipliers.
- **Outputs:**  
  Individual-level risk fields appended to the synthetic population.

### **Component 3: Insurance Coverage Transition**
- **Purpose:**  
  To simulate how many and which individuals lose Medicaid coverage and become uninsured under user-specified policy scenarios, stratified by demographics.
- **Inputs:**  
  Tract-level baseline Medicaid enrollment rates.  
  Policy loss rates and optional demographic multipliers.
- **Outputs:**  
  Updated insurance status fields (e.g., Medicaid, uninsured, private) for each individual.

### **Component 4: Screening Access Modeling**
- **Purpose:**  
  Estimate the probability each individual receives recommended colorectal cancer screening, based on real-world participation rates stratified by insurance status and demographics.
- **Inputs:**  
  Demographic/risk-modified insurance status for each synthetic person. Public survey parameters (typically from BRFSS or NHIS) for screening by subgroup.
- **Outputs:**  
  Individual screening status or probability.

### **Component 5: Health Economics Projection**
- **Purpose:**  
  To estimate both the medical consequences (cancer incidence, staging shift) and macroeconomic outcomes (aggregate cost, potential cost-savings) for the scenario of interest.
- **Inputs:**  
  Projected cancer risk, screening, and staging for all individuals.  
  SEER-Medicare parameters for cost by stage and stage-shift benefit of screening.
- **Outputs:**  
  Aggregated tract, subregion, and scenario-level cost, case, and impact projections.

---

## File-by-File Implementation Map

| Filename                             | Purpose and Role                                                                       |
|--------------------------------------|----------------------------------------------------------------------------------------|
| ccrat_model_demographic.py           | Implements the demographic microsimulation engine. Defines the synthetic population distribution. |
| ccrat_model_risk_assessment.py       | Contains CCRAT risk computations and applies demographic multipliers to individuals.    |
| ccrat_model_insurance.py             | Calculates insurance loss transitions; applies policy and demographic adjustment rates. |
| ccrat_model_screening.py             | Assigns screening probabilities based on insurance and demographic status.              |
| ccrat_model_economics.py             | Computes cancer incidence, staging, and full cost/benefit economic projections.         |
| ccrat_model_validation.py            | Includes statistical validation tools for outputs and model health diagnostics.         |
| ccrat_model_utils_config.py          | Configuration management, including paths, tolerances, and YAML integration.            |
| ccrat_model_utils_data_loader.py     | Loads raw demographic and parameter data. Provides class to load census tract table and inputs.  |
| ccrat_model_parameters.json          | Stores all model parameters: CCRAT baselines, multipliers, cost data, loss rates, stage distributions, etc. |
| data_validation_summary.json         | Provides dataset-level checks such as population sums, demographic min/max, descriptive stats.   |
| example_individual_risk.py           | Example script for running individual-level risk calculations and displaying outputs.   |

---

## Incorporating Data Sources

This section identifies public data resources for each required model input, and how those sources should be “attached” to the pipeline, replacing or supplementing synthetic/demo data as appropriate.

### **1. Census Tract Demographics**
- **Public Source:**  
  U.S. Census Bureau American Community Survey (ACS) 5-year estimates  
  - [data.census.gov](https://data.census.gov/)
  - ACS API (tables B01001, B27001, C17002, etc.)
- **How to use:**  
  Download relevant tables by tract for study region.  
  These CSVs are loaded by `ccrat_model_utils_data_loader.py` and should be formatted to match columns expected by `ccrat_model_demographic.py` (population, poverty, Medicaid, etc.).

---

### **2. CCRAT Risk Parameters**
- **Public Source:**  
  NCI Colorectal Cancer Risk Assessment Tool documentation and publications  
  (parameters encoded in `ccrat_model_parameters.json`)
  - [NCI CCRAT Portal](https://ccrisktool.cancer.gov/calculator.html)
  - Freedman et al., 2009, J Clin Oncol (formula)
- **How to use:**  
  Copy risk baselines and multipliers into `ccrat_model_parameters.json` (or update via script).
  Ensure parameter JSON reflects any new literature or calibration applied.

---

### **3. Insurance Coverage Transitions & Medicaid Loss**
- **Public Source:**  
  - ACS B27001 (Health Insurance by age/sex/race/poverty by tract)
  - MACPAC Unwinding Data Dashboards ([macpac.gov](https://www.macpac.gov/))
  - Kaiser Family Foundation (KFF) Medicaid Unwinding Tracker ([kff.org](https://www.kff.org/medicaid/))
  - National Health Interview Survey (NHIS) microdata ([CDC NHIS](https://www.cdc.gov/nchs/nhis/))
- **How to use:**  
  Download disenrollment/loss rates stratified by available categories (age, race, income).  
  Update `ccrat_model_parameters.json` with demographic-specific adjustment multipliers.
  `ccrat_model_insurance.py` reads these multipliers and applies them when simulating transitions.

---

### **4. Screening Participation / Access**
- **Public Source:**  
  - CDC Behavioral Risk Factor Surveillance System (BRFSS) [cdc.gov/brfss](https://www.cdc.gov/brfss/)
  - National Health Interview Survey (NHIS)
- **How to use:**  
  Extract (or compute) screening participation rates by insurance, age, race, income using microdata or state summary.  
  Store in `ccrat_model_parameters.json` under relevant keys (e.g., `"screening_rates_by_insurance"`, `"demographic_adjustments"`).
  `ccrat_model_screening.py` will map insurance+demographics to screening probability.

---

### **5. Cancer Staging and Costs**
- **Public Source:**  
  - SEER-Medicare Linked Database ([seer.cancer.gov](https://seer.cancer.gov/))
  - Published cost analyses (Warren et al., JNCI Monographs; Mariotto et al., Med Care)
- **How to use:**  
  Use national mean/median cost by cancer stage (I–IV) and typical stage distribution for screened/unscreened.  
  Parameterize in `ccrat_model_parameters.json` under `"treatment_costs"` and `"stage_distributions"`.
  Used for calculation of tract and population cost projections in `ccrat_model_economics.py`.

---

## Example: Attaching New Data to the Code Base

- To update insurance transition stratifiers, download new MACPAC or ACS files and parse to get adjustment rates by age/race/income.  
  Paste those into the corresponding section of `ccrat_model_parameters.json`, e.g.:
 
"demographic_adjustments": {
"age": {"45_49": 1.2, ...},
"race": {"black": 1.15, ...}
}

- To update screening rates, repeat with latest BRFSS/NHIS prevalence data, updating in the parameter JSON:
"screening_rates_by_insurance": {"medicaid": 0.48, "uninsured": 0.29, ...}


- To add or alter costs/stage distributions, use SEER/Medicare or other studies—update `"treatment_costs"` and `"stage_distributions"` in the JSON accordingly.

- All code modules that require new/updated data will automatically use the refreshed parameters file when rerun.

---

## Directory and File Structure Example

colon_cancer/
├── ccrat_model_demographic.py # (Component 1)
├── ccrat_model_risk_assessment.py # (Component 2)
├── ccrat_model_insurance.py # (Component 3)
├── ccrat_model_screening.py # (Component 4)
├── ccrat_model_economics.py # (Component 5)
├── ccrat_model_validation.py
├── ccrat_model_utils_config.py
├── ccrat_model_utils_data_loader.py
├── ccrat_model_parameters.json # Model parameters from public sources
├── data_validation_summary.json # Summary diagnostic
├── example_individual_risk.py

text

---

## Install

**Install dependencies** (`pip install pandas numpy`)
