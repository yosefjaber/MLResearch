import pandas as pd
import numpy as np

# Reads the data converts to UTF-8 and removes all blank rows
df = pd.read_csv("data/original-data.csv", encoding="latin-1")
before = len(df)
df.to_csv("data/original-data-utf.csv", encoding="utf-8", index=False)
original = pd.read_csv("data/original-data-utf.csv")
clean = original.replace('---', np.nan)
clean = clean.dropna()

def clean_row_off_tolorence(df,name_of_row,tolerance):
    df_row = df[name_of_row]
    values_to_drop = []
    
    for i in range(len(df_row)):
        if i > 0 and abs(df_row[i-1]-df_row[i]) >= tolerance:
            values_to_drop.append(i-1)
            values_to_drop.append(i)
    
    for i in range(len(values_to_drop)):
        print(f"df[i-1]: {df[values_to_drop[i]-1]}, df[i]: {df[values_to_drop[i]]}, diff: {abs(df[values_to_drop[i]-1] - df[values_to_drop[i]])}")
    
    df = df.drop(values_to_drop)

print(clean.head())
clean.to_csv('data/clean-data.csv', index=False)
