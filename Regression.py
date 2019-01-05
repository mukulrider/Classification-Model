import datetime

df_training  =  df[df['txn_date']<'2018-06-11']
from patsy import dmatrices, Treatment
y, X = dmatrices('churn_perc ~ nuts_perc+android_perc+wap_perc+website_perc+ios_perc+nb_perc+non_promo_txn_perc+prepaid_perc+other_business_subtype_perc+ruts_perc+dth_perc+broadband_perc+wallet_only_perc+pg_wallet_perc+p2p_perc+upi_perc+dc_perc+pg_only_perc+postpaid_perc+electricity_perc+cc_perc', df_training, return_type = 'dataframe')

df=df[(df['column'].isin([20,40,50])) | (df['column']>=75)]
df.loc[df['column']==20,'class']=20
df.loc[df['column'].between(75,100),'class']=75


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/mg10825/Desktop/mukul/Analysis Files/NUTS Model/dau_regression_data.csv", sep=',',header=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

features = [x for x in list(df.columns) if x not in ('churn_perc','txn_date','euts_perc',
            'promo_txn_perc','others_pg_perc','windows_perc','dau_count',
            'pg_only_perc','android_perc','wallet_only_perc','wallet_upi_perc',
            'prepaid_perc','dth_perc','broadband_perc','electricity_perc','website_perc','cashback_burned',
            'upi_perc','p2p_perc','nb_perc','upi_only_perc')]
target='churn_perc'

X = df[df['txn_date'].between('2018-01-01','2018-06-15')][features]
Y = df[df['txn_date'].between('2018-01-01','2018-06-15')][target]
X_oot = df[df['txn_date'].between('2018-06-16','2018-07-31')][features]
Y_oot = df[df['txn_date'].between('2018-06-16','2018-07-31')][target]

import statsmodels.api as sm
#from scipy import stats

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

#print(est2.pvalues)
#print(est2.params)
#print(est2.bse)
chi_square = (est2.params/est2.bse)**2

print(chi_square)

# Calculating VIF for Variable to Eliminate Variables having Multicollinearity 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif["features"] = X2.columns

print(vif.round(1))

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
Y_pred = clf.predict(X)
Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)
#del(X_oot['const'])
Y_pred_oot = clf.predict(X_oot)
#print (Y_poly_pred)
print("Train Data Mean Squared Error:",mean_squared_error(y_train,Y_pred_train))
print("Train Data Variance Score:",r2_score(y_train,Y_pred_train))
print("Test Data Mean Squared Error:",mean_squared_error(y_test,Y_pred_test))
print("Test Data Variance Score:",r2_score(y_test,Y_pred_test))
print("OOT Data Mean Squared Error:",mean_squared_error(Y_oot,Y_pred_oot))
print("OOT Data Variance Score:",r2_score(Y_oot,Y_pred_oot))
#print("OOT Data Variance Score:",clf.score(X_oot,Y_oot))

from sklearn.ensemble import GradientBoostingRegressor 
gbrt=GradientBoostingRegressor(n_estimators=2500,
                               max_depth=20,
                               learning_rate=0.001,
                               min_samples_leaf=6,
                               max_features=10,
                               random_state=42) 
gbrt.fit(X_train, y_train) 
print ("R-squared for Train:", gbrt.score(X_train, y_train))
print ("R-squared for Test:", gbrt.score(X_test, y_test))

Y_pred = gbrt.predict(X)
Y_pred_train=gbrt.predict(X_train)
Y_pred_test=gbrt.predict(X_test)
Y_pred_oot=gbrt.predict(X_oot)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Train Data MAPE:",mean_absolute_percentage_error(y_train,Y_pred_train))
print("Test Data MAPE:",mean_absolute_percentage_error(y_test,Y_pred_test))
print("OOT Data MAPE:",mean_absolute_percentage_error(Y_oot,Y_pred_oot))


## Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X_train)
X_test_ = poly.fit_transform(X_test)
clf.fit(X_,y_train)
y_test=clf.predict(X_test_)

df2 = df[df['txn_date'].between('2018-01-01','2018-06-15')]
df2['predicted'] = Y_pred
df2['residual'] = df2['churn_perc']-df2['predicted']

plt.hist(df2['residual'])
df2.to_csv("C:/Users/mg10825/Desktop/mukul/Analysis Files/Dau Regression Analysis GBM Test Results (1Jan-15June).csv")

#OOT Data for Validation
df3 = df[df['txn_date'].between('2018-06-16','2018-07-31')]
df3['predicted'] = Y_pred_oot
df3.to_csv("C:/Users/mg10825/Desktop/mukul/Analysis Files/Dau Regression GBM OOT Validation Results (16June-19July).csv")

# Read Data for Which Churn is to be forecasted
df_oot = pd.read_csv("C:/Users/mg10825/Desktop/mukul/Analysis Files/NUTS Model/dau_regression_data_11July_19Aug.csv", sep=',',header=0)
X_oot=df_oot[features]
Y_pred_oot=gbrt.predict(X_oot)
df_oot['predicted']=Y_pred_oot
