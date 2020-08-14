# CreditRisk-Analysis Using Machine Learning Models
## Background

Using data from LendingClub; a peer-to-peer lending services, several machine learning models were used to assess, their relative performance to accurately predict credit risk.
The analysis was done by cleaning the data and splitting it into Target variable(Loan status) and features(columns that served as input values to help predict the target value).
Credit risk being an unbalanced classification problem(The ratio of good loans to risky loans being very high ), a variety of sampling techniques were used to get a balanced and representative sample, to train and evaluate models.

Following sampling techniques and models were used:

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
    
 ## Resources
 
 ### Data:
 - [LoanStats2019Q1.csv](https://github.com/Muzznah/CreditRisk-Analysis_ML/blob/master/Challenge/Module-17-Challenge-Resources/__MACOSX/Module-17-Challenge-Resources/._LoanStats_2019Q1.csv)
 
 ### Software:
    -Python 3.8.2, Jupyter Notebook 6.0.3, numpy 1.19.1, pandas 1.0.5, scikit-learn 0.23.2, and imblearn 0.7.0
 ## Analysis
 
 The analysis process was broken down into
    - Preprocessing Data(Encoding, Splitting, Scaling)
    - Training and Testing
    - and Deploying the Models
    
 At the end of every model deployment, a Balanced Accuracy Score, Confusion Matrix and Imbalanced Classification Report was generated to assess the performace of that model.   
 The following table shows a summary comparison across models (For code detail check [resampling.ipynb](https://github.com/Muzznah/CreditRisk-Analysis_ML/blob/master/CreditRisk-MLM/credit_risk_resampling-final-revised.ipynb) and [ensemblers.ipynb](https://github.com/Muzznah/CreditRisk-Analysis_ML/blob/master/CreditRisk-MLM/credit_risk_ensemble-final.ipynb):
 

![](https://github.com/Muzznah/CreditRisk-Analysis_ML/blob/master/CreditRisk-MLM/ML-Comparison.png)

From the above table we can see that all the models had a perfect precision score of 1.0, a decent recall score(ranging from 0.76 to 0.94), and decent to great F1 score (ranging from 0.86 to 0.97)for detecting low-risk loans. This means that there were very few False Positives or False Negatives. However, the objective here is to ensure, that maximum number of high-risk loans are accurately flagged.

The precision values for high-risk loans across all models was very low (ranging from 0.02 to 0.09) indicating that there was a high number of False Positives( low-risk loans being labled as high-risk), in contrast the recall value for high-risk loans was decent to great (ranging from 0.7 to 0.92), indicating a low count of False Negatives (high-risk loans being labled as low-risk). The overall F1 score for high-risk loans was again very low( ranging from 0.04 to 0.16).

**The Best Performing Model:**

Overall looking at all the metrics, the model that performed the best was, 'Easy Ensemble Adaboost
Classifier', with highest F1 score (for high-risk) of 0.16, highest F1 score (for low-risk) of 0.97 and the best accuracy score of 0.93.

**The Lowest Performing Model:**

The model that didnt fare all that well realtive to others was undersampling with clustercentroids on logistic regression model. with lowest F1 of 0.4 (for high-risk), lowest F1 of 0.86 (for low-risk) and the second lowest balanced accuracy score of 0.82.

**The Best Performing Model minus Ensemble:**

the model that performed the best was, 'SMOTE', with relatively high F1 score (for high-risk) of 0.07,  relatively high F1 score (for low-risk) of 0.93 and a good balanced accuracy score of 0.84.

**Balanced Random Forest Classifier's top 3 contributing feature variables were:**

        -   0.079, 'total_rec_prncp'
        -   0.059, 'total_pymnt'
        -   0.056, 'total_pymnt_inv'
        
## Conclusion

The "Easy Ensemble Adaboost", was the best performing model in this analysis and is hence recommended. However, we need to keep in mind that due to its high recall value and low precision value, it will have high False Positives, meaning many low-risk loans might get labled as high-risk as well, and will require additional time, effort and resources to vet them out. So keeping this in mind the recommended model can be used as an initial screening which can be further inspected.
