import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


data = {
'TransactionID': [1, 2, 3, 4, 5],
'Items': [
['Bread', 'Milk'],
['Bread', 'Diaper', 'Beer', 'Eggs'],
['Milk', 'Diaper', 'Beer', 'Coke'],
['Bread', 'Milk', 'Diaper', 'Beer'],
['Bread', 'Milk', 'Diaper', 'Coke']
]
}
df = pd.DataFrame(data)
print("Initial Data:\n", df)

df_items = df['Items'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
print("\nOne-Hot Encoded Data:\n", df_items)

frequent_itemsets = apriori(df_items, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules)

for _, row in rules.iterrows():
    print(f"\nRule: {set(row['antecedents'])} -> {set(row['consequents'])}")
print(f"Support: {row['support']:.2f}")
print(f"Confidence: {row['confidence']:.2f}")
print(f"Lift: {row['lift']:.2f}")