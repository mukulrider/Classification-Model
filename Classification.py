# Classification-Model

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
df = pd.read_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/RUTS Model/mukul_ruts_propensity_data_jun1_july10.csv", sep=',',header=0,index_col=0,usecols=['col1','col2','col3','col4'])



#df1 = pd.read_sql_query("SELECT * FROM tempload.mukul_cust_reactivation_propensity_data_val_final;", engine)
df1 = pd.read_sql_query("SELECT * FROM tempload.mukul_ruts_propensity_data_jun1_july10 WHERE grade_overall IS NOT NULL LIMIT 10000000;", con)
df=df1.copy()
con.close()
df=df.loc[df['txn_count']>2]

#df1.to_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/RUTS Model/mukul_ruts_propensity_data_jun1_july10.csv")

# Calculating Aggregated Value and renaming Column
d=df.groupby(['pg_mode','platform','txn_type']).size().reset_index(name='counts')
print(d)

# Column Values Frequency Distribution
df['column_name'].value_counts().nlargest(10)/df['column_name'].count()
df[df['column_name1']=='Prepaid']['column_name2'].value_counts().nlargest(10).plot.bar()

# Dummifying Categorical Columns
column_values = {'Prepaid' : 1, 'Postpaid' : 2, 'DTH':3, 'Electricity':4, 'Broadband':5, 'ONLINE':6}
df.replace({'column': column_values}, inplace=True)

# Changing Datatype to Integer
df["column_name"]=pd.to_numeric(df["column_name"],downcast='integer')

## Check for Columns having NULL Values
df.columns[df.isna().any()].tolist()

for col in df.columns:
 df[col] = df[col].fillna(0,inplace=True)
 
df['grade_overall'] = df['grade_overall'].fillna('Unknown')
 
#df.loc[df['grade_overall'].isin(['NA']),'grade_overall']='Unknown'
#df=df[df['first_txn_date'].between('2018-04-01','2018-05-15')]

# pg_mode_values = {'AMEX':'CC','DINR':'CC'}
# df.replace({'pg_mode':pg_mode_values}, inplace=True)
#for column in df.columns:
#    df[column].fillna(0, inplace=True)
#d=df.groupby(['pg_mode','platform','txn_type']).size().reset_index(name='counts')

df_bivariate = df.groupby('churn').mean()[['platform_Android']]
df_bivariate.plot.line()

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

from  collections import OrderedDict

ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,class_weight='balanced',
                               max_features="sqrt",
                               criterion = 'entropy', random_state = 42))
    ,("RandomForestClassifier, max_features=22",
        RandomForestClassifier(warm_start=True, max_features='log2',class_weight='balanced',
                               oob_score=True,
                               random_state=42))
    ,("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,class_weight='balanced',
                               oob_score=True,
                               random_state=42))
]

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 50
max_estimators = 150

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1,20):
        clf.set_params(n_estimators=i)
        clf.fit(x_validate, y_validate)
        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))
        print(i, clf.oob_score_)



# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")

kfold = model_selection.KFold(n_splits=5, random_state=42)
ABClassifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=42)
ABClassifier.fit(X_train,y_train)
results = model_selection.cross_val_score(ABClassifier, X_train, y_train, cv=kfold)
print(results.mean())

kfold = model_selection.KFold(n_splits=5, random_state=42)
GBClassifier = GradientBoostingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, learning_rate=1.0, random_state=42)
GBClassifier.fit(X_train,y_train)
results = model_selection.cross_val_score(GBClassifier, X_train, y_train, cv=kfold)
print(results.mean())

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import precision_score

joblib.dump(RFClassifier, 'C://Users/fcmg10825/Desktop/mukul/NUTS Model/nuts_churn_prediction_new.pkl') 
import  imp
from sklearn.externals import joblib
from time import gmtime, strftime
loaded_model = joblib.load('/home/pi_analytics/Mukul/NutsModel/nuts_churn_prediction_new.pkl')

print("Model Imported!")
y_pred = loaded_model.predict(df2)

df['churn_pred']=y_pred
df['churn_prob']=loaded_model.predict_proba(df2)[:,1]

conditions = [
    (df['churn_prob'] >=0) & (df['churn_prob'] <=0.155847),
    (df['churn_prob'] >0.155847) & (df['churn_prob'] <=0.232601),
    (df['churn_prob'] >0.232601) & (df['churn_prob'] <=0.346943),
    (df['churn_prob'] >0.346943) & (df['churn_prob'] <=0.456311),
    (df['churn_prob'] >0.456311) & (df['churn_prob'] <=0.594372),
    (df['churn_prob'] >0.594372) & (df['churn_prob'] <=0.720649),
    (df['churn_prob'] >0.720649) & (df['churn_prob'] <=0.862170),
    (df['churn_prob'] >0.862170) & (df['churn_prob'] <=0.949210),
    (df['churn_prob'] >0.949210) & (df['churn_prob'] <=0.983249),
    (df['churn_prob'] >0.983249)]
deciles = [1,2,3,4,5,6,7,8,9,10]
df['decile'] = np.select(conditions, deciles)

df['first_txn_date']=pd.to_datetime(df['first_txn_date'], format='%Y-%m-%d', errors='ignore')
df['model_run_date']=df['first_txn_date']+pd.DateOffset(days=4)
df['inserted_time']=strftime("%Y-%m-%d %H:%M:%S", gmtime())

df.loc[df['decile'] == 1,'action_date']=df['model_run_date']+pd.DateOffset(days=15)
df.loc[df['decile'].between(2,5),'action_date']=df['model_run_date']+pd.DateOffset(days=7)
df.loc[df['decile'].between(6,10),'action_date']=df['model_run_date']+pd.DateOffset(days=3)

from sqlalchemy import create_engine

df3 = df[['ims_id','first_txn_date','churn_pred','churn_prob','decile','model_run_date','action_date','inserted_time']]
df3.to_csv("/data/pi_data/Mukul/nuts_data/nuts_churn_action_date_new.csv",index=False)
