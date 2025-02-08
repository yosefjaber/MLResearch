import pandas as pd
import numpy as np

df = pd.read_csv("data/original-data.csv", encoding="latin-1")

df.to_csv("data/original-data-utf.csv", encoding="utf-8", index=False)

original = pd.read_csv("data/original-data-utf.csv")

clean = original.replace('---', np.nan)

clean = clean.dropna()

print(clean.head())

clean.to_csv('data/clean-data.csv', index=False)
