# CreditRisk-Analysis Using Machine Learning Models
## Background

Using data from LendingClub; a peer-to-peer lending services, several machine learning models were used to assess, their relative performance to accurately predict credit risk.
The analysis was done by cleaning the data and splitting it into Target variable(Loan status) and features(columns that served as input values to help predict the target value).
Credit risk being an unbalanced classification problem(The ratio of good loans to risky loans being very high ), the following sampling techniques were used to train and evaluate models:

**1- Oversampling**</br>

    -Random Oversampling
    -SMOTE
**2- Undersampling**</br>

    -Clustercentroids
**3- Combination (of under and over) Sampling**</br>

    - SMOTEEN
**4- Ensemble learners**

    -Balanced Random Forest Classifier
    -Easy Ensemble AdaBoost Classifier
    
 ## Analysis
 
 The analysis process was broken down into
    - Preprocessing Data(Encoding, Splitting, Scaling)
    - Training and Testing
    - and Deploying the Models
    
 At the end of every model deployment, a Balanced Accuracy Score, Confusion Matrix and Imbalanced Classification Report was generated to assess the performace of that model.   
 The following table shows a summary comparison across models.
 

![](https://github.com/Muzznah/CreditRisk-Analysis_ML/blob/master/CreditRisk-MLM/ML-Comparison.png)
