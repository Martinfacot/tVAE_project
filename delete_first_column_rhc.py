import pandas as pd

df = pd.read_csv('rhc.csv')

df = df.drop(df.columns[0], axis=1)

df.to_csv('rhc.csv', index=False)