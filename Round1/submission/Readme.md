
# Hackathon Project Documentation

## Project Overview

This documentation outlines the key aspects of our hackathon project, focusing on advanced techniques, the outcomes of machine learning (ML) and neural network experiments, and insights derived from data analysis and visualisations.

---

## 1. Advanced Techniques: Choices and Benefits

### Techniques Used:
- **Feature Engineering**: Implemented advanced feature extraction techniques, including polynomial transformations and normalisation, to enhance input quality for models.
- **Neural Network Architecture**: Utilised deep learning with layers optimised for non-linear patterns, including dropout layers to prevent overfitting.
- **Hyperparameter Tuning**: Automated optimisation using grid search and early stopping to determine the best model parameters.
- **Validation Strategy**: Adopted stratified k-fold cross-validation for balanced assessment of imbalanced datasets.

### Benefits:
- The use of **feature engineering** improved signal-to-noise ratio, allowing models to focus on meaningful data patterns.
- Neural networks excelled in capturing **non-linear relationships** and dependencies in high-dimensional data.
- Hyperparameter tuning ensured **robust generalisation** across unseen data.
- Cross-validation provided a **comprehensive performance assessment**, reducing the risk of overfitting.

---

## 2. Machine Learning (ML) Challenges

### Observations:
- Classical ML models such as logistic regression and decision trees showed suboptimal performance.
- Inability to model complex non-linear relationships resulted in low accuracy.
- High variance in validation results indicated overfitting despite regularisation efforts.

### Root Causes:
- Dataset complexity required models capable of representing intricate patterns.
- Limited capacity of ML algorithms to scale with high-dimensional input data.

---

## 3. Success of Neural Networks

### Performance Highlights:
- Achieved a significant accuracy improvement (~20%) over traditional ML models.
- Successfully addressed class imbalance with techniques such as **weighted loss functions**.
- Captured intricate patterns through deep layers and activation functions (ReLU).

### Key Innovations:
- **Dropout Layers**: Prevented overfitting by randomly deactivating neurons during training.
- **Batch Normalisation**: Improved convergence speed and stabilised learning.
- **Dynamic Learning Rates**: Enhanced adaptability during gradient descent.

---

## 4. Insights from Data Analysis

### Feature Insights:
- **onus_attribute_1**: Identified as **credit limit**, influencing high-risk outcomes.
- **transaction_attribute**: Strongly associated with payments to merchants, signalling potential financial strain.
- **Plotting Correlations**: Revealed key attributes affecting risk levels (e.g., credit utilisation).

### Visualisation Highlights:
1. **Risk Distribution**:
   - Plots showed ~15% of users classified as high-risk.
   - Key factors included transaction density and repayment delays.

2. **Model Performance**:
   - Precision-Recall curves highlighted improved recall rates for high-risk groups.
   - Neural network's AUC score outperformed classical models consistently (~0.85 vs. ~0.65).

3. **Feature Importance**:
   - Heatmaps visualised correlations between features and high-risk labels.
   - High-weighted attributes included credit card utilisation and spending ratios.

### Validation Metrics:
- Stratified sampling ensured balanced testing:
  - **High-Risk Population**: 15%.
  - **Validation Accuracy**: Neural network achieved **85%**.
  - **Class Imbalance**: Addressed using SMOTE and cost-sensitive learning.

---

## Conclusion

Our hackathon project demonstrated the power of advanced neural networks in solving complex classification problems. While ML models fell short in capturing non-linear relationships, neural networks excelled due to their flexibility and depth. Insights gained from visualisations and data analysis underscored the importance of feature engineering and class balancing in high-stakes decision-making systems.

### Next Steps:
- Further optimise neural architectures.
- Explore hybrid models combining ML and deep learning.
- Integrate additional real-world datasets for robustness.
