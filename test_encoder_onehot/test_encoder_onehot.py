import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'personal_packages')))


from personal_packages.data_loading import read_csv
from personal_packages.CategoricalEncoder import CategoricalEncoder



# this file is used to test the CategoricalEncoder class and data_loading.py

# load data
data, discrete_columns = read_csv(
    csv_filename="data_test.csv",
    meta_filename="metadata_test.json",
    header=True
)

print("original data:")
print(data)
print("\nColumns discrete :", discrete_columns)

# encode data
encoder = CategoricalEncoder()
encoder.fit(data, discrete_columns=discrete_columns)
encoded_data = encoder.transform(data)

print("\nEncoded data :")
print(pd.DataFrame(encoded_data))

# decode data
decoded_data = encoder.inverse_transform(encoded_data)

print("\ndecoded datga :")
print(decoded_data)

# check if the original and decoded data are the same
print("\nCheck :")
print("original data are the same ?", 
      data.reset_index(drop=True).equals(decoded_data.reset_index(drop=True)))
