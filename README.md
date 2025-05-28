# Vehicular Hazard Prediction

This repository implements time-series forecasting and classification models for predicting vehicular hazard attacks using the [VeReMiAP dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/R09EWA).

It includes LSTM and GRU models for univariate hazard forecasting, as well as a stacked model that combines LSTM regression outputs with logistic regression for improved binary classification.

---

## 🚀 Features

- **LSTM Forecasting**  
  Time-series hazard attack prediction using a Long Short-Term Memory model.

- **GRU Forecasting**  
  Forecasting hazard signals using a Gated Recurrent Unit (GRU) model — a simpler and computationally efficient alternative to LSTM.

- **Stacked Model (LSTM + Logistic Regression)**  
  Integrates LSTM output with raw features to classify hazard attacks using logistic regression.

- **Performance Metrics**  
  Evaluates models using R² score, MAE, F1 score, accuracy, and confusion matrix.

- **Visualization**  
  Time-based plots for actual vs. predicted hazard occurrences and classifications.

- **Export Options**  
  Predictions and metrics can be saved to CSV files; models can be stored for later reuse.

---

## 📂 File Structure

| File                                  | Description                                                         |
|---------------------------------------|---------------------------------------------------------------------|
| `lstm_hazard_forecast_model.py`      | LSTM model for time-series forecasting of hazard attacks            |
| `gru_hazard_forecast_model.py`       | GRU model for hazard attack forecasting *(to be added)*             |
| `stacked_hazard_prediction_model.py` | Combines LSTM regression with logistic classification (stacked model) |
| `README.md`                           | Project documentation                                               |

---

## 🧠 Requirements

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn seaborn

python lstm_hazard_forecast_model.py

python gru_hazard_forecast_model.py

python stacked_hazard_prediction_model.py
