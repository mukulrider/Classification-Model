# Classification-Model

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:05:18 2018

@author: fcmg10825
"""
# Garbage Collection
import gc
 
collected = gc.collect()
 
# Prints Garbage collector 
# as 0 object
print("Garbage collector: collected",
          "%d objects." % collected)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import psycopg2
df = pd.read_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/RUTS Model/mukul_ruts_propensity_data_jun1_july10.csv", sep=',',header=0,index_col=0)

con=psycopg2.connect(dbname= 'walletdb', host='kiwi.cd2pip2qjan8.ap-south-1.redshift.amazonaws.com', 
port= '5439', user= 'analyticsuser', password= 'Thisispasswordfornewkiwianalyticsuser1!')


#df1 = pd.read_sql_query("SELECT * FROM tempload.mukul_cust_reactivation_propensity_data_val_final;", engine)
df1 = pd.read_sql_query("SELECT * FROM tempload.mukul_ruts_propensity_data_jun1_july10 WHERE grade_overall IS NOT NULL LIMIT 10000000;", con)
df=df1.copy()
con.close()
df=df.loc[df['txn_count']>2]

#df1.to_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/RUTS Model/mukul_ruts_propensity_data_jun1_july10.csv")

## Check for Columns having NULL Values
df.columns[df.isna().any()].tolist()
df['last_txn_promo_amount'] = df['last_txn_promo_amount'].fillna(0)
df['grade_overall'] = df['grade_overall'].fillna('Unknown')
#df.loc[df['grade_overall'].isin(['NA']),'grade_overall']='Unknown'

#df[['campaign_amount']] = df[['campaign_amount']].apply(pd.to_numeric,errors='ignore')
#df.loc[df['last_txn_product'].isin(['DONATION','Datacard Postpaid','FC_CREDIT','Metro','Water','windows','Datacard',
#'Landline','Gas','PAID_COUPONS','Google','Broadband']),'last_txn_product']='others'
#df.drop(['campaign_date'], axis = 1, inplace = True)

df=pd.get_dummies(df)
df.drop(['last_txn_product_DTH','last_txn_platform_android','grade_overall_Unknown',
         'grade_overall_low_value','dc_txn_perc'], axis = 1, inplace = True)
df.head()

import pandas as pd
#from sklearn import svm
#from sklearn import model_selection
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.metrics import confusion_matrix
#from sklearn.externals import joblib
print('Libraries Imported')

features = [x for x in list(df.columns) if x not in ('is_reactivated')]
#features=list((list(df.columns))-(['class']))
target='is_reactivated'
print(features)

# Generate the training set.  Set random_state to be able to replicate results.
X=df[features]
Y=df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

## GridSearch CV to select the best parameters
param_grid = { 
    'n_estimators': [50],
    'min_samples_leaf':[500,750],
    'max_features': [22,24,26,28,30,32],
    'n_jobs':[-1],
 #   'max_depth' : [20],
    'criterion' :['entropy'],
    'min_impurity_decrease': [0]
}

from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier(random_state=42,class_weight='balanced')
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5,verbose=1)
CV_rfc.fit(X_test, y_test)

print("Best parameters: {}".format(CV_rfc.best_params_))

## Training The Random Forest Classifier
RFClassifier = RandomForestClassifier(n_estimators = 100,max_features=32,criterion = 'entropy',bootstrap=True,min_samples_leaf=500,n_jobs = -1, random_state = 42,class_weight='balanced')
RFClassifier.fit(X_train, y_train)

y_pred = RFClassifier.predict(X_test)
print(pd.crosstab(y_test, y_pred, rownames=['Actual Class'], colnames=['Predicted Class']))


## Plot Confusion Matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not-Reactivated','Reactivated'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not-Reactivated','Reactivated'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()


## Random Forest Classifier Accuracy Score
from sklearn.metrics import accuracy_score
#print(X_test[:,-1])
#print(y_test)
print("Accuracy Score:  {:02.2f} %".format(accuracy_score(y_test,y_pred)*100))

## Random Forest Classifier Precision, Recall, Fscore, Support
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test,y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

## Random Forest Variable Importances
feature_list=list(features)
# Get numerical feature importances
importances = list(RFClassifier.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance*100, 3)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


X_test['churn_prob']=RFClassifier.predict_proba(X_test)[:,1]
X_test['is_reactivated']=y_test
X_test['deciles']=pd.qcut(X_test['churn_prob'], 10, labels=range(1,11), retbins=False, duplicates='raise')


a=X_test.groupby(['deciles']).agg({'days_since_last_txn': 'mean','cust_age':'mean','promo_txn_count':'mean','campaign_amount':'mean','last_txn_promo_amount':'mean','txn_count':'mean','last_txn_promo_use':'sum','virtual_txn_count':'mean','atv': 'mean',
                'last_txn_amount':'mean','last_txn_platform_wap':'sum','is_reactivated':'sum'})
print(a)

b=X_test.groupby(['deciles']).size()
print(b)
# ROC-AUC Curve
# calculate the fpr and tpr for all thresholds of the classification
from sklearn.metrics import roc_curve, auc
probs = RFClassifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

import scikitplot as skplt
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb = nb.fit(X_train, y_train)
y_probas = nb.predict_proba(X_test)
skplt.metrics.plot_ks_statistic(y_test, y_probas)
plt.show()

import statsmodels.api as sm
#from scipy import stats

X2 = sm.add_constant(X_train)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

df['const']=1
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]
vif["features"] = X_test.columns
print(vif.round(1))

## Calculate VIF and drop variables with higher VIF than threshold iteratively
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=10.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=10.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X
    
transformer = ReduceVIF()
X = transformer.fit_transform(X_test, y_test)

X.head()

from sklearn.metrics import roc_curve, auc
# calculate the fpr and tpr for all thresholds of the classification
probs = RFClassifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()