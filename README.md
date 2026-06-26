# Fetal Health Classification from Cardiotocography Data

A machine learning project for classifying fetal health conditions from Cardiotocography (CTG) data using multiple classification algorithms.

## Team Members

- Surianandhan Sridhar
- Pattan Sameera Hussainy

---

## Project Overview

This project predicts fetal health status using Cardiotocography (CTG) measurements. Three machine learning models are trained and compared:

- Logistic Regression
- XGBoost
- Multi-Layer Perceptron (MLP Neural Network)

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training
- Performance evaluation
- Cross-validation
- Feature importance analysis
- Automatic generation of publication-quality visualizations

---

## Dataset

**Dataset:** Fetal Health Classification

Source:
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

### Classes

| Label | Description |
|-------|-------------|
| 1 | Normal |
| 2 | Suspect |
| 3 | Pathological |

The dataset contains **2,126 CTG records** with **21 numerical features** and one target variable (`fetal_health`).

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Project Structure

```
.
├── fetal_health_classification.py
├── fetal_health.csv
├── model_summary.csv
├── figures/
│   ├── fig1_class_distribution.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_feature_distributions.png
│   ├── fig4_confusion_matrices.png
│   ├── fig5_accuracy_comparison.png
│   ├── fig6_f1_comparison.png
│   ├── fig7_mlp_loss_curve.png
│   ├── fig8_feature_importance.png
│   └── fig9_cross_validation.png
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Surianandhan/fetal-health-classification.git
```

Move into the project folder:

```bash
cd fetal-health-classification
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Running the Project

Place the dataset (`fetal_health.csv`) in the project directory.

Run:

```bash
python fetal_health_classification.py
```

The program will:

- Load the dataset
- Perform Exploratory Data Analysis
- Train all three models
- Evaluate model performance
- Save all figures
- Export a summary table (`model_summary.csv`)

---

## Machine Learning Pipeline

1. Load Dataset
2. Data Exploration
3. Feature Engineering
4. Train-Test Split
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Cross Validation
9. Visualization
10. Result Summary

---

## Models Implemented

### Logistic Regression

- Multinomial classification
- Standardized features
- Baseline linear classifier

### XGBoost

- Gradient Boosting Decision Trees
- Feature importance analysis
- High predictive performance

### MLP Neural Network

- Hidden Layers: 128 → 64 → 32
- ReLU activation
- Adam optimizer
- Early stopping enabled

---

## Generated Outputs

The program automatically creates the following figures:

1. Class Distribution
2. Feature Correlation Heatmap
3. Feature Distribution by Class
4. Confusion Matrices
5. Model Accuracy Comparison
6. Per-Class F1 Score Comparison
7. MLP Training Loss Curve
8. XGBoost Feature Importance
9. Cross-Validation Accuracy Box Plot

Additionally:

- `model_summary.csv`

contains the Accuracy, Precision, Recall and F1-score for all models.

---

## Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- 5-Fold Cross Validation

---

## Results

The project compares the performance of:

- Logistic Regression
- XGBoost
- MLP Neural Network

Performance is evaluated using weighted classification metrics and visual comparisons to determine the best-performing model.

---

## Future Improvements

- Hyperparameter optimization
- Additional ensemble models
- Explainable AI (SHAP/LIME)
- ROC and Precision-Recall Curves
- Model deployment using Flask or Streamlit
- Real-time fetal health prediction interface

---

## License

This project is intended for educational and academic purposes.

---

## Acknowledgements

- Kaggle for providing the dataset
- Scikit-learn developers
- XGBoost developers
- Open-source Python community
