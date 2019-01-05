import pandas as pd
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df1 = pd.read_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/UPI Clustering/upi_txn_agg_data_with_dts_data.csv", sep=',',header=0)

len(df1[df1['upi_txns']>=9])
len(df1)
df1=df1[df1['cust_age_wrt_reg_date'].notnull()]

len(df1)
len(df1[df1['ims_id'].isna()])

df1=df1[df1['ims_id'].notnull()]
df=df1.copy()

len(df[df['promo_txn_perc'].isna()])
len(df[df['p2p_txn']==0])

len(df[df['p2p_ats'].isna()])
len(df[df['p2m_ats'].isna()])
len(df[df['p2m_txn']==0])
df.columns

print(df.columns[df.isna().any()].tolist())
#df['days_since_last_upi_txn'] = df['days_since_last_upi_txn'].fillna(-999)
df['p2p_promo_txn_perc']=df['p2p_promo_txn']*100.0/df['p2p_txn']
df['p2m_promo_txn_perc']=df['p2m_promo_txn']*100.0/df['p2m_txn']
df['promo_txn_perc'] = df['promo_txn_perc'].fillna(0)
df['p2p_ats'] = df['p2p_ats'].fillna(0)
df['p2m_ats'] = df['p2m_ats'].fillna(0)
df['p2p_cb_tpv'] = df['p2p_cb_tpv'].fillna(0)
df['p2m_cb_tpv'] = df['p2m_cb_tpv'].fillna(0)
df['p2p_promo_txn_perc']=df['p2p_promo_txn_perc'].fillna(0)
df['p2m_promo_txn_perc']=df['p2m_promo_txn_perc'].fillna(0)
df=df.fillna(0)


for col in df1.columns:
    percentiles = df1[col].quantile([0.01,0.99]).values
    df1[col]=[df1[col] <= percentiles[0]] 
    df1[col][df1[col] >= percentiles[1]] = percentiles[1]


print(df[df['p2p_promo_txn_perc']==0].head())
#perc_txns = np.percentile(df['txns'], 99)
#perc_p2p_gmv = np.percentile(df['p2p_gmv'], 99)
#perc_p2m_gmv = np.percentile(df['p2m_gmv'], 99)

df= df.drop(['bank_pub_slb','bank_priv_slb','payment_bank','other_banks'], axis = 1)

print(np.mean(df['txns']))
print(np.std(df['txns']))

perc_txns1 = np.mean(df['all_upi_txns'])-3*np.std(df['all_upi_txns'])
perc_txns2 = np.mean(df['all_upi_txns'])+3*np.std(df['all_upi_txns'])


perc_p2p_gmv1 = np.mean(df['p2p_gmv'])-3*np.std(df['p2p_gmv'])
perc_p2p_gmv2= np.mean(df['p2p_gmv'])+3*np.std(df['p2p_gmv'])


perc_p2m_gmv1 = np.mean(df['p2m_gmv'])-3*np.std(df['p2m_gmv'])
perc_p2m_gmv2 = np.mean(df['p2m_gmv'])+3*np.std(df['p2m_gmv'])

perc_p2p_txn = np.mean(df['p2p_txn'])+3*np.std(df['p2p_txn'])
perc_p2m_txn = np.mean(df['p2m_txn'])+3*np.std(df['p2m_txn'])

perc_p2p_cb = np.mean(df['p2p_cb'])+3*np.std(df['p2p_cb'])
perc_p2m_cb = np.mean(df['p2m_cb'])+3*np.std(df['p2m_cb'])

print(perc_txns2,perc_p2p_gmv2,perc_p2m_gmv2,perc_p2p_txn,perc_p2m_txn,perc_p2p_cb,perc_p2m_cb)


##Values Between 3 S.D.(99% values)
df=df.loc[(df['all_upi_txns']<=perc_txns2) & (df['p2p_gmv']<=perc_p2p_gmv2) & (df['p2m_gmv']<=perc_p2m_gmv2)
& (df['p2p_txn']<=perc_p2p_txn) & (df['p2m_txn']<=perc_p2m_txn)& (df['p2p_cb']<=perc_p2p_cb) & (df['p2m_cb']<=perc_p2m_cb)]


df1=df1.loc[(df1['all_upi_txns']<=perc_txns2) & (df1['p2p_gmv']<=perc_p2p_gmv2) & (df1['p2m_gmv']<=perc_p2m_gmv2)
& (df1['p2p_txn']<=perc_p2p_txn) & (df1['p2m_txn']<=perc_p2m_txn)& (df1['p2p_cb']<=perc_p2p_cb) & (df1['p2m_cb']<=perc_p2m_cb)]

df.sort_values(['user_id'], ascending=[True])
df1=df.copy()

print(len(df1),len(df))
df= df.drop(['user_id','ims_id'], axis = 1)
df= df.drop(['days_since_last_upi_txn'], axis = 1)
## Values Between 2.5 S.D.
#df=df.loc[(df['txns']<=80.2) & (df['p2p_gmv']<51473.6) & (df['p2m_gmv']<4196.2)]

## Values Between 2 S.D. (95.4% Values)
df=df.loc[(df['txns']<=65.26) & (df['p2p_gmv']<41494.18) & (df['p2m_gmv']<3472.05)]

##df1.shape
#df1['service_region'].unique()
#df1['service_provider_merchant'].unique()
#
#df1.loc[df1['service_region'].isin(['Delhi','Mumbai','Chennai','Kolkata']),'service_region']='Metro'
#df1.loc[df1['service_region'].isin(['Karnataka','Kerala','Tamil Nadu','Andhra Pradesh']),'service_region']='South'
#df1.loc[df1['service_region'].isin(['Bihar','Rajasthan','Uttar Pradesh (E)','Uttar Pradesh (W)','Madhya Pradesh','JK']),'service_region']='Tier3'
#df1.loc[df1['service_region'].isin(['Assam','Orissa','Haryana','Punjab']),'service_region']='Tier2'
#df1.loc[df1['service_region'].isin(['West Bengal','Gujarat','UTTARANCHAL','Maharashtra','Himachal Pradesh','Delhi']),'service_region']='Tier1'
#
#
#
#df1.loc[df1['service_provider_merchant'].isin(['Airtel','Airtel Postpaid']),'service_provider_merchant']='Airtel'
#df1.loc[df1['service_provider_merchant'].isin(['Idea','Idea Postpaid']),'service_provider_merchant']='Idea'
#df1.loc[df1['service_provider_merchant'].isin(['Vodafone','Vodafone Postpaid']),'service_provider_merchant']='Vodafone'
#df1.loc[df1['service_provider_merchant'].isin(['JIO','JIO PostPaid']),'service_provider_merchant']='Reliance'
#df1.loc[df1['service_provider_merchant'].isin(['BSNL','Bsnl Postpaid','MTNL Mumbai','MTNL Delhi']),'service_provider_merchant']='Govt'
#df1.loc[df1['service_provider_merchant'].isin(['Tata Docomo GSM','Uninor','Tata Docomo CDMA','Docomo Postpaid-GSM','Docomo Postpaid-CDMA','Aircel']),'service_provider_merchant']='Idea'
len(data1)
df=df.sample(n=2000000, replace=False, random_state=42)
#a=df1.groupby(['service_region']).agg({'txn_count':'sum','txn_amount':'mean','count_prepaid':'sum','count_postpaid':'sum', 'count_electricity':'sum', 'count_dth':'sum',
#       'count_other_billpay':'sum', 'count_ondeck':'sum', 'count_offdeck':'sum', 'count_pg_only':'sum',
#       'count_pg_wallet':'sum'})
#a.to_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/Customer Profitability Clustering/service_region_analysis_results.csv",index=True)
#



#data['service_provider_merchant'] = data['service_provider_merchant'].fillna('Unknown')
#data['service_region'] = data['service_region'].fillna('Unknown')

#df=pd.get_dummies(data)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

print(df.columns)
data1=pd.DataFrame(data=scaled_data,columns=df.columns)

print("Length df1:",len(df1),"Length df:",len(df))

sse = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, max_iter=300,random_state=42,n_jobs=-1).fit(data1)
    data1['clusters'] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k,max_iter=300,random_state=42,n_jobs=-1).fit(df)
    label = kmeans.labels_
    sil_coeff = silhouette_score(df, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
    
kmeans = KMeans(n_clusters=5,max_iter=300,random_state=42).fit(data1)
df['clusters'] = kmeans.labels_

data1.head(10)
df1.head(10)
df1['clusters']=data1['clusters']
df['clusters']=data1['clusters']
df2=scaler.inverse_transform(data1)
df2=pd.DataFrame(df2)
df2.head(10)

a=df.groupby(['clusters']).agg({'cust_age_wrt_reg_date':'mean',
            'upi_txns':'mean','upi_promo_txn_perc':'mean','p2p_txn':'mean','p2m_txn':'mean', 'receive_txn':'mean',
            'p2p_promo_txn':'mean','p2m_promo_txn':'mean',
            'p2p_gmv':'mean','p2m_gmv':'mean','p2p_ats':'mean','p2m_ats':'mean',
            'p2p_cb':'mean','p2m_cb':'mean','distinct_user_transfer':'mean','distinct_user_receive':'mean',
            'p2p_freq':'mean','p2m_freq':'mean','upi_freq':'mean',
            'total_txn':'mean','promo_txn_perc':'mean','cc_txns':'mean','dc_txns':'mean','nb_txns':'mean','upi_txns':'mean',
            'distinct_category':'mean','distinct_other_category':'mean',
            'ats':'mean','tpv':'mean','cb':'mean','cb_to_tpv':'mean','pg_txn':'mean','prepaid':'mean',
            'postpaid':'mean','dth':'mean','online':'mean','electricity':'mean','pg_only':'mean',
            'upi_only':'mean','wallet_only':'mean','wallet_upi':'mean','pg_wallet':'mean',
            'distinct_banks':'mean', 
#            'bank_pub_slb':'sum','bank_priv_slb':'sum','payment_bank':'sum','other_banks':'sum',
            'payment_banks_txn':'mean','pvt_banks_txn':'mean','pub_banks_txn':'mean','other_banks_txn':'mean'
            })
#a=df.groupby(['clusters']).mean()
a.to_csv("C:/Users/fcmg10825/Desktop/mukul/Analysis Files/UPI Clustering/cluster_analysis_results_with_dts_c5_v3.csv",index=False)

print(df.head)
b=df.groupby(['clusters']).size()
print(b)
