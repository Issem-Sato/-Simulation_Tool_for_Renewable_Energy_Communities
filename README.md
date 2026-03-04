# CER Simulator

Streamlit-based simulator for Renewable Energy Communities (CER) and collective self-consumption.
It supports scenario analysis of energy flows (production/consumption/sharing) and economic-financial evaluation, including BESS-related sensitivity analyses.

## Thesis
The full explanation is in file thesis.pdf

**Title:** CER Simulator  
**Candidate:** Filip Juren  
**Supervisor:** Luigi De Paoli  
**Academic Year:** 2024/25

## Main features
- Modular architecture: Streamlit orchestration layer (`src/cer_app`) + core computation modules (`src/cer_core`)
- Session/scenario/run persistence for reproducibility and traceability
- Renewable production modeling (PV / wind) and data acquisition workflows
- Consumption profile generation and aggregation
- Energy balancing and shared energy computation (hourly)
- Economic evaluation with tariffs/incentives assumptions and scenario comparisons

## Installation and Run (Windows)

1. Install **Anaconda Navigator**: https://www.anaconda.com/products/navigator  
2. Extract the project `.zip` to your **Desktop** (or any folder you prefer).  
3. Open the `cer-simulator` folder.  
4. Run **as Administrator**: `setup_structure.bat`  
   - This creates the environment and downloads the required libraries.  
5. Run **as Administrator**: `start_cer.bat`  
   - The simulator will open in your browser.
