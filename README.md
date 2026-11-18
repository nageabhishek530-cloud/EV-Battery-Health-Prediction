# EV Battery Health & Remaining Useful Life Prediction

An AI-powered web application that predicts the State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion batteries using Machine Learning.

## Project Overview

This project uses machine learning algorithms to analyze battery discharge data and estimate battery health and remaining useful life.

The application provides a web-based dashboard where users can enter battery operating parameters and receive predictions.

## Objectives

- Predict battery State of Health (SOH).
- Estimate Remaining Useful Life (RUL).
- Compare the performance of three machine learning algorithms.
- Visualize battery capacity degradation.
- Provide an interactive web dashboard for predictions.

## Dataset

**NASA Lithium-Ion Battery Aging Dataset**

The project uses four experimental battery datasets:

- B0005
- B0006
- B0007
- B0018

The dataset contains battery discharge information, including:

- Cycle number
- Capacity
- Voltage
- Current
- Temperature

The dataset contains experimental lithium-ion cells, not complete commercial EV battery packs.

## Machine Learning Algorithms

Three machine learning algorithms are used:

1. **Random Forest Regressor**
2. **XGBoost Regressor**
3. **Support Vector Regression (SVR)**

These algorithms are evaluated for battery health and remaining useful life prediction.

## Prediction Targets

### 1. State of Health (SOH)

SOH represents the remaining battery capacity compared with its initial capacity.

Formula:

SOH = (Current Capacity / Initial Capacity) × 100

### 2. Remaining Useful Life (RUL)

RUL estimates the number of discharge cycles remaining until the battery reaches the project's selected 70% capacity threshold.

This threshold is a project-defined assumption.

## Model Performance

### SOH Prediction Results

| Algorithm | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 2.0913 | 2.6583 | 0.9226 |
| XGBoost | 2.3130 | 2.9815 | 0.9084 |
| SVR | 3.3803 | 4.4137 | 0.7659 |

### RUL Prediction Results

| Algorithm | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 49.0619 | 51.1430 | -1.1831 |
| XGBoost | 52.9006 | 54.6867 | -1.5428 |
| SVR | 38.3819 | 43.2950 | -0.3641 |

**Note:** RUL prediction has limitations because only two of the four experimental cells reach the selected 70% capacity threshold in the recorded data. The negative R² values indicate weak generalization on the battery-wise evaluation.

## Technologies Used

- Python
- Flask
- NumPy
- Pandas
- SciPy
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib
- HTML
- CSS

## Project Structure

```text
EV-Battery-Health-Prediction/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── static/
│   ├── images/
│   ├── capacity_graph.png
│   └── model_comparison.png
│
├── templates/
│   └── index.html
│
├── app.py
├── prepare_dataset.py
├── create_targets.py
├── train_models.py
├── train_rul.py
├── save_models.py
├── create_graph.py
├── create_comparison_chart.py
├── inspect_data.py
│
├── soh_model.pkl
├── rul_model.pkl
├── rul_scaler.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd EV-Battery-Health-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open the application

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## Application Features

- Interactive EV battery analytics dashboard
- SOH prediction
- RUL prediction
- Battery condition assessment
- Model performance comparison
- Battery capacity degradation visualization
- Prediction history

## Limitations

- The dataset contains four experimental lithium-ion cells.
- Only two cells reach the selected 70% capacity threshold.
- RUL predictions have limited generalization due to the small number of valid battery-life observations.
- Results should not be treated as certified predictions for real-world EV battery packs.

## Future Improvements

- Use a larger and more diverse battery dataset.
- Improve RUL prediction accuracy.
- Explore deep learning approaches such as LSTM.
- Deploy the application to a cloud platform.
- Integrate real-time battery monitoring data.

## Conclusion

This project demonstrates how machine learning can be applied to lithium-ion battery health monitoring and remaining useful life estimation through an interactive web application.

It also compares three machine learning algorithms and highlights the challenges of predicting battery life using limited experimental data.

---

**Developed as an academic Machine Learning Project.**