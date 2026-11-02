# Battery Health Prediction System Using ML

A ready-to-run educational project that predicts a battery's State of Health (SOH) from simulated input parameters.

## Important note
This project uses **synthetic / simulated data**, not measurements from real batteries. Model metrics only describe performance on held-out synthetic records. This is not a real EV BMS and must not be used for vehicle diagnostics, safety decisions, or battery protection.

## Included
- `data/battery_dataset.csv` — 5,000 synthetic records
- `train_model.py` — trains and evaluates the model
- `models/battery_soh_model.joblib` — pre-trained model
- `reports/model_evaluation.json` — test metrics and limitations
- `app.py` — interactive Streamlit app
- `requirements.txt` — Python packages
- `Run_Project.bat` — Windows launch helper

## Easiest way on Windows
1. Extract the ZIP file.
2. Make sure Python is installed and available from Command Prompt (`python --version` or `py --version`).
3. Double-click `Run_Project.bat`.
4. The first launch installs the required packages; this needs an internet connection.
5. The app should open in your browser. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Manual run
Open Command Prompt in this folder and run:
```bash
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
If `python` does not work, try `py` instead.

## Model inputs
- `Cycle_Count`
- `Avg_Temperature_C`
- `Avg_Cell_Voltage_V`
- `Avg_Current_A`
- `Depth_of_Discharge_Percent`
- `Charge_Rate_C`

Target: `SOH_Percent`

The input ranges and relationships are designed for a simulated single-cell educational example. They are not calibrated to a particular EV pack.

## Re-training
If you want to regenerate training results from the included dataset, run:
```bash
python train_model.py
```
The script overwrites the saved model and evaluation report.
