import pandas as pd

data_path = "/home/milan/Desktop/Elfak_praksa/Kitsune-py/archive/03-02-2018.csv"

df = pd.read_csv(data_path, low_memory=False)

print(df["Label"].value_counts())