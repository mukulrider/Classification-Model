df = pd.read_csv("C:/Users/Desktop/mukul/Analysis Files/Market Basket Analysis/market_basket_analysis_loan_data_all.csv", sep=',',header=0)


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from collections import OrderedDict


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = df.applymap(encode_units)

frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

df2=rules[ (rules['lift'] > 1) & (rules['confidence'] >= 0.35) ]
len(df2)
