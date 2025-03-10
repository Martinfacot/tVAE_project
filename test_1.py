import pandas as pd
from data_loading import read_csv
from CategoricalEncoder import CategoricalEncoder


# this file is used to test the CategoricalEncoder class and data_loading.py

# load data
data, discrete_columns = read_csv(
    csv_filename="data.csv",
    meta_filename="metadata.json",
    header=True
)

print("original data:")
print(data)
print("\nColumns discrete :", discrete_columns)

# encode data
encoder = CategoricalEncoder()
encoder.fit(data, discrete_columns=discrete_columns)
encoded_data = encoder.transform(data)

print("\nDonnées encodées :")
print(pd.DataFrame(encoded_data))

# decode data
decoded_data = encoder.inverse_transform(encoded_data)

print("\nDonnées décodées :")
print(decoded_data)

# check if the original and decoded data are the same
print("\nCheck :")
print("original data are the same ?", 
      data.reset_index(drop=True).equals(decoded_data.reset_index(drop=True)))
