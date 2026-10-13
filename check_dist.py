import pandas as pd

df = pd.read_csv('train.csv')
missing_values = df.isnull().sum()
less = missing_values[missing_values < 1000].index
over = missing_values[missing_values >= 1000].index

df2 = df.drop(columns=over)

numeric_features = df2[less].select_dtypes(include=['number']).columns
for c in df2[less].select_dtypes(include=['object']).columns:
    df2[c] = df2[c].fillna(df2[c].mode()[0])

df2[numeric_features] = df2[numeric_features].fillna(df2[numeric_features].median())

Q1 = df2[numeric_features].quantile(0.25)
Q3 = df2[numeric_features].quantile(0.75)
IQR = Q3 - Q1
condition = ~((df2[numeric_features] < (Q1 - 1.5 * IQR)) | (df2[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)

before = df2.copy()
after = df2.loc[condition].copy()

print('rows_before=', len(before))
print('rows_after=', len(after))
print('mean_before')
print(before[numeric_features].mean().round(3).to_string())
print('mean_after')
print(after[numeric_features].mean().round(3).to_string())
print('std_before')
print(before[numeric_features].std().round(3).to_string())
print('std_after')
print(after[numeric_features].std().round(3).to_string())
print('min_before')
print(before[numeric_features].min().round(3).to_string())
print('min_after')
print(after[numeric_features].min().round(3).to_string())
print('max_before')
print(before[numeric_features].max().round(3).to_string())
print('max_after')
print(after[numeric_features].max().round(3).to_string())
