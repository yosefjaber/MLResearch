import pandas as pd
import numpy as np

# Reads the data converts to UTF-8 and removes all blank rows
df = pd.read_csv("data/original-data.csv", encoding="latin-1")
print(len(df))
df.to_csv("data/original-data-utf.csv", encoding="utf-8", index=False)
original = pd.read_csv("data/original-data-utf.csv")
clean = original.replace('---', np.nan)
clean = clean.dropna()
clean = clean.reset_index(drop=True)
before = len(clean)
values_to_drop = []

def clean_row_off_condition(df, name_of_row, func, values_to_drop):
    df_row = df[name_of_row]
    
    for i in range(len(df_row)):
        # Use positional indexing with .iloc
        if i > 0 and func(float(df_row.iloc[i])):
            values_to_drop.append(i)
            #print(df_row.iloc[i])
        
    return values_to_drop

def clean_row_off_tolorence(df,name_of_row,tolerance,values_to_drop):
    df_row = df[name_of_row]
    
    for i in range(len(df_row)):
        if i > 0 and abs(float(df_row.iloc[i-1])-float(df_row.iloc[i])) >= tolerance:
            values_to_drop.append(i-1)
            values_to_drop.append(i)
            #print(f"{float(df_row.iloc[i])} is i, {float(df_row.iloc[i-1])} is i - 1, diff is {abs(float(df_row.iloc[i-1])-float(df_row.iloc[i]))}")
    return values_to_drop

#

# values_to_drop = clean_row_off_condition(clean, "Hot Water System : Hot Water Supply Temperature Local (°F)",lambda x : x < 115, values_to_drop)

values_to_drop = clean_row_off_tolorence(clean, "AHU-01 VAV : Hot  Water Coil Flow Meter Local (gpm)", 0.50, values_to_drop)
values_to_drop = clean_row_off_tolorence(clean, "AHU-01 VAV : Supply Fan Air Flow Local (cfm)", 250, values_to_drop)
values_to_drop = clean_row_off_condition(clean, "Hot Water System : Hot Water Supply Temperature Local (°F)",lambda x : x < 100, values_to_drop)

clean = clean.drop(values_to_drop)

print(clean)
after = len(clean)
print(f"Change: {before - after}")
sum = 0
for i in range(len(clean["Hot Water System : Hot Water Supply Temperature Local (°F)"])):
    sum += float(clean["Hot Water System : Hot Water Supply Temperature Local (°F)"].iloc[i])
print(sum/len(clean["Hot Water System : Hot Water Supply Temperature Local (°F)"]))
clean.to_csv('data/clean-data.csv', index=False)

#Delete anything below 100T of in water, also remove delta of 0.5 gpm, remove deltas of 250cfm, remove 2 goofy points