# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:50:41 2019

@author: 44919
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
loan_cleaned = pd.read_csv('C:/Users/44919/Data science job/Data Challenge/loan_cleaned.csv')
# remove features Only has 1 value
one_value_features = []

for f in loan_cleaned:
    if len(loan_cleaned[f].value_counts()) == 1:
        one_value_features.append(f)

loan_cleaned.drop(one_value_features, axis=1, inplace=True)
# Convert string to float
loan_cleaned['int_rate'] = loan_cleaned['int_rate'].apply(lambda x: float(x[:-1]))
# decrease levels in zip_code
loan_cleaned['zip_code'] = loan_cleaned['zip_code'].apply(lambda x: x[0]+x[-1])
# delete sub_grade
loan_cleaned.drop(['sub_grade'], axis=1, inplace=True)
# delete addr_state
loan_cleaned.drop(['addr_state'], axis=1, inplace=True)
# get dummy
# extract categorical variables
dummy_columns = [x for x in loan_cleaned if loan_cleaned[x].dtype != 'float64']
# get dummies
loan_cleaned = pd.get_dummies(loan_cleaned, columns = dummy_columns)
from sklearn.model_selection import train_test_split
x = loan_cleaned[loan_cleaned.columns.difference(['int_rate'])]
y = loan_cleaned['int_rate'].values
# Split data into train and test (80% & 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
# normalize features
from sklearn.preprocessing import StandardScaler
# initialize a scaler object
scaler = StandardScaler()
# transform training set
x_train_std = scaler.fit_transform(x_train)
# the same transform for test set
x_test_std = scaler.transform(x_test)
from sklearn.linear_model import RidgeCV
## ridge regression
alphas_ridge = np.arange(1e-6, 1e-3, 5e-6)
# initialize a model object
RidgeReg = RidgeCV(alphas = alphas_ridge, store_cv_values=True)
# train model
RidgeReg.fit(x_train_std, y_train)
# get optimal alpha 
# draw coefficent graph
coef = pd.Series(RidgeReg.coef_, index = x.columns)
importance_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.figure(figsize=(12,8))
importance_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Regression")
plt.show()
# gradient boosting 
# initialize model
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, )
# fit model
gb.fit(x_train_std, y_train)
# draw importance graph
featureImportance = gb.feature_importances_
features = pd.DataFrame()
features['features'] = x_train.columns.values
features['importance'] = featureImportance
features.sort_values(by=['importance'],ascending=False,inplace=True)
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
plt.xticks(rotation=90)
sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h", color="#34495e")
plt.title("Feature Importance in Gradient Boosting", fontsize=20)
plt.show()



