# Bankruptcy Prediction in Polish Companies Using a Neural Network from Scratch

## Executive Summary
This project implements a neural network from scratch to predict corporate bankruptcy in Polish companies. The dataset is highly imbalanced, with bankrupt companies forming a minority. The solution demonstrates real-world machine learning reasoning, handling class imbalance with SMOTE, optimizing training using mini-batch gradient descent, and prioritizing recall for risk detection. This approach balances computational efficiency and practical relevance for financial risk assessment.

---

## Problem Statement
- Predict whether a company will go bankrupt based on financial ratios.
- Dataset contains 64 financial features for ~7,000 Polish companies.
- Severe class imbalance (~95% non-bankrupt, ~5% bankrupt) makes traditional accuracy metrics misleading.

---

## Solution
- Built a **feedforward neural network from scratch** using Python and NumPy.
- Implemented **forward propagation, backpropagation, and gradient descent manually**.
- Applied **SMOTE** to training data to balance minority-class representation.
- Used **mini-batch gradient descent** for computational efficiency and stability.
- Explored **decision threshold tuning** to optimize recall of bankrupt companies.

---

## Impact / Significance
- Model captures **55% of bankrupt companies** at default threshold (0.5), significantly improving minority-class detection.
- Highlights **trade-offs between accuracy and risk sensitivity**, critical for financial decision-making.
- Demonstrates how to handle **imbalanced datasets**, a common real-world challenge.

---

## Methodology
1. **Data Preprocessing**
   - Missing values imputed with column mean.
   - Features normalized using Min-Max scaling.
   - Train-test split with 80-20 ratio.
   - SMOTE applied **only to training set**.
2. **Neural Network Architecture**
   - Input: 64 features
   - Hidden layer: 16 neurons, ReLU activation
   - Output layer: 1 neuron, Sigmoid activation
   - Loss: Binary Cross-Entropy
   - Optimizer: Mini-batch Gradient Descent (batch size = 64)
   - Epochs reduced to prevent overfitting and ensure Colab stability.
3. **Evaluation**
   - Confusion matrix
   - Precision, recall, and F1-score
   - Threshold tuning for business risk sensitivity
4. **Visualization**
   - Training loss over epochs plotted for convergence analysis.

---

## Skills and Technologies
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib  
- **Concepts:** Neural networks, backpropagation, mini-batch gradient descent, class imbalance handling, threshold tuning, binary classification  
- **Tools:** Google Colab

---

## Results
- Test Accuracy: ~57% (threshold 0.5)
- Minority-class recall: 55% (detecting bankrupt companies)
- Precision (bankrupt): 6% — expected trade-off for risk sensitivity
- Confusion matrix confirms meaningful minority-class detection
- Threshold tuning shows impact on precision-recall trade-offs
- Loss curve shows stable convergence; included in notebook for transparency

---

## Colab Notebook
- Full implementation and outputs available on Google Colab:  
[Open Colab Notebook](https://colab.research.google.com/drive/165Ssz2g1uEVKlxmlzieqdeI5Y9GY_otn?usp=sharing)

> *Note:* Epochs were reduced from 5000 to 1000 to avoid Colab crashing while still achieving convergence. Mini-batch training ensured memory efficiency.

---

## Next Steps / Future Work
- Explore **hyperparameter tuning** (hidden layer size, learning rate) for improved minority-class performance.
- Implement **advanced optimizers** (Adam, RMSProp) for faster convergence.
- Extend to **multi-class financial risk prediction**.
- Deploy the model in a **real-time risk assessment pipeline**.
