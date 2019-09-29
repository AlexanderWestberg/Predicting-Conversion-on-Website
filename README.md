# Predicting-Conversion-on-Website
Using Machine Learning to predict conversions on a website
Mostly have focused on Ensemble Learning models, due to other regular models such as Logistic Regression and KNN did not do a good job.

I focused on the f1 score to evaluate the models. Because Some models such as The AdaBoost had a very high precision but a bad recall, and 
and other had high recall and low precision. So the f1 score seemed to solve that problem quite good. 

The model that had the highest f1 score was RandomForestClassifier, with an approximate 0.734 F1_Score.

Up to this point I have mostly focused on optimizing and finding a good model, but not so much on feature selection and feature importance. I will probabliy try on focus on that more in the future.

Data Source: 
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
