# Data

The project uses the German Credit dataset (`credit-g`) from OpenML.

The dataset is downloaded automatically by `src/challenger_model.py`, so the raw dataset is not stored in this repository.

The project also creates a simulated `dpd_30_plus` feature for the data-leakage demonstration. This variable is generated inside the Python script and is not part of the original dataset.
