# Turbofan RUL Prediction using XGBoost and Time-Series Windowing

This project predicts the Remaining Useful Life (RUL) of turbofan jet engines using machine learning on the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset.

The project is inspired by the existing work **“A Machine Learning Approach for Turbofan Jet Engine Predictive Maintenance”** and extends the baseline approach by using XGBoost regression with time-series windowing for improved RUL prediction.

## Project Overview

Predictive maintenance helps estimate when an engine may require inspection, monitoring, or replacement. In this project, turbofan sensor data is used to predict engine RUL and support maintenance decision-making.

The system includes:

* RUL calculation from engine cycle data
* Preprocessing of operational settings and sensor readings
* Selection of relevant operational and sensor features
* Sliding-window feature engineering to capture degradation trends
* XGBoost regression for RUL prediction
* Model evaluation using R², RMSE, and MAE
* Health-state classification for maintenance interpretation
* Confidence-based decision support

## Dataset

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

The dataset contains multivariate time-series readings from simulated turbofan engines, including operational settings, sensor measurements, and engine run-to-failure cycles. It is widely used for predictive maintenance and Remaining Useful Life estimation tasks.

## Methodology

The project follows this workflow:

1. Load and preprocess the NASA C-MAPSS turbofan dataset.
2. Calculate Remaining Useful Life using engine cycle information.
3. Apply RUL clipping to stabilize model training.
4. Select relevant operational and sensor features.
5. Generate time-series windows from engine sensor readings.
6. Train an XGBoost regression model for RUL prediction.
7. Evaluate the model using regression metrics.
8. Convert predicted RUL into interpretable health states.
9. Map health states into maintenance decisions.

## Results

| Work                | Model               |     R² |  RMSE |    MAE |
| ------------------- | ------------------- | -----: | ----: | -----: |
| Gupta et al. (2025) | Random Forest       |  0.789 |  19.0 | ~14–15 |
| Proposed Work       | XGBoost + Windowing | 0.8517 | 15.43 |  11.59 |

Higher R² indicates better explanatory performance, while lower RMSE and MAE indicate lower prediction error.

## Health-State Mapping

The predicted RUL values are mapped into maintenance-oriented health states:

| Predicted RUL Range | Health State | Maintenance Meaning            |
| ------------------- | ------------ | ------------------------------ |
| RUL > 80            | Healthy      | Normal operation               |
| 30 < RUL ≤ 80       | Warning      | Monitor closely                |
| RUL ≤ 30            | Replace      | Replacement attention required |

## Tech Stack

* Python
* Pandas
* NumPy
* scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook / Google Colab

## Project Status

This project is currently in progress. Future improvements include extended benchmarking, additional validation on other C-MAPSS subsets, and further improvement of health-state classification and maintenance decision support.

## Reference

Gupta et al. (2025), **“A Machine Learning Approach for Turbofan Jet Engine Predictive Maintenance”**, Procedia Computer Science, Elsevier.
