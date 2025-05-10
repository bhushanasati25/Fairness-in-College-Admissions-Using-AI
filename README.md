# Fairness in College Admissions AI

## Overview

This project implements a fairness-aware AI system for college admissions that identifies and mitigates biases in the admissions process. Using the College Scorecard dataset from the U.S. Department of Education, the system evaluates different machine learning models, quantifies their bias using multiple fairness metrics, and applies bias mitigation techniques to improve fairness while maintaining predictive performance.

## Problem Statement

College admissions processes often exhibit biases that disadvantage underrepresented groups like minorities and low-income students. AI systems used in admissions can unintentionally amplify these biases, perpetuating inequality in access to higher education. This project aims to develop a fairness-aware AI system for college admissions that:

1. Identifies biases in admission decisions
2. Mitigates these biases using state-of-the-art techniques
3. Provides transparency through comprehensive visualizations and dashboards
4. Balances predictive accuracy with fairness metrics

## Features

- **Data preprocessing**: Handles missing values and normalizes features from the College Scorecard dataset
- **Multiple ML models**: Implements Logistic Regression, Decision Trees, Random Forest, and XGBoost
- **Fairness metrics**: Evaluates Demographic Parity, Disparate Impact, and Equalized Odds
- **Bias mitigation techniques**:
  - Reweighting: Adjusts training weights to balance outcomes across groups
  - Threshold Optimization: Applies different decision thresholds for different demographic groups
  - Fairness Constraints: Removes demographic features from the model
- **Comprehensive visualizations**: 
  - Interactive dashboards using Plotly
  - Performance comparison charts
  - Fairness metrics visualization
  - Feature importance analysis
  - 3D visualization of decision space
- **Synthetic applicant generation**: Creates diverse applicant profiles for prediction testing

## Installation

```bash
# Clone the repository
git clone https://github.com/bhushanasati25/Fairness-in-College-Admissions-Using-AI.git
cd fairness-college-admissions

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
xgboost==1.7.5
plotly==5.14.1
```

## Data

This project uses the College Scorecard dataset from the U.S. Department of Education, which includes:
- Student demographics (race, gender, income level)
- Admission criteria (SAT, GPA, etc.)
- University acceptance rates
- Graduation & post-graduation outcomes

Place the `Most-Recent-Cohorts-Institution.csv` file in the project directory to run the code.

## Usage

```python

# To use specific components:
from fairness_college_admissions import load_and_preprocess_data, train_and_evaluate_models

# Load and preprocess data
data = load_and_preprocess_data("Most-Recent-Cohorts-Institution.csv")

# Prepare data for modeling
X_train, X_test, y_train, y_test, protected_train, protected_test, X_train_scaled, X_test_scaled, scaler = prepare_data_for_modeling(data)

# Train and evaluate models
models, results_df, fairness_df, predictions, probability_scores = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, protected_test)
```

## Methodology

1. **Data Collection & Preprocessing**: Extract key features (SAT, GPA, demographic data)
2. **Model Training**: Train multiple models for admission prediction
3. **Fairness Analysis**: Apply fairness metrics to evaluate bias
4. **Bias Mitigation**: Implement fairness-aware AI techniques
5. **Dashboard Development**: Create interactive visualizations 

## Results

The project compares different models and bias mitigation techniques, typically finding:

- XGBoost achieves the highest accuracy but often shows bias
- Threshold optimization provides the best balance between fairness and accuracy
- Removing demographic features improves some fairness metrics but reduces model performance
- Reweighting the training data helps balance outcomes across demographic groups

Interactive dashboards (saved as HTML files) provide a comprehensive view of model performance and fairness metrics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- U.S. Department of Education for the College Scorecard dataset
- The fairness in the machine learning research community for algorithms and metrics
- Open-source ML libraries including scikit-learn, XGBoost, and Plotly
