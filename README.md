# SMU FTEN Bayesian Institutional Research Lab

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Packaged datasets
- SMU_649_CLEAN_COHORT.csv — all unique valid FTEN students for descriptive profiling.
- SMU_246_TRAIN_OBSERVED_OUTCOME.csv — students with genuine matched HEMIS outcomes.
- SMU_246_CORE_MODEL_READY.csv — compact model-ready matrix.
- SMU_DATA_QUALITY_SUMMARY.csv — reconciliation checks.

## Interpretation
This is an explanatory/exploratory Bayesian model. It is not intended for automated adverse decisions.
Academic outcomes are never imputed. HEMIS credits used to define the outcome are excluded from predictors.
