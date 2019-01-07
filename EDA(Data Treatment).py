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
 
## Outlier Detection Plots 
i = 'cb_txn_count'
q75, q25 = np.percentile(df.cb_txn_count.dropna(), [75 ,25])
iqr = q75 - q25
 
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min,max)
 
plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df[i].min(), df[i].max()*1.1)
plt.axvline(x=min)
plt.axvline(x=max)
 
ax = df[i].plot(kind='kde')
 
plt.subplot(212)
plt.xlim(df[i].min(), df[i].max()*1.1)
sns.boxplot(x=df[i])
plt.axvline(x=min)
plt.axvline(x=max)
 
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
