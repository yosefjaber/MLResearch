import pandas as pd
import numpy as np

# Reads the data converts to UTF-8 and removes all blank rows
original_1 = pd.read_csv("data/original-data-1-utf.csv")

original_2 = pd.read_csv("data/original-data-2-utf-head.csv")
original_2 = original_2.drop(["Date", "Time", "Time Zone"], axis=1)
new_column_order = ['Discharge Temp(F)', 
                   'Water Temp(F)',
                   'Water Flow Rate(gpm)',
                   'Intake Air Temp(F)',
                   'Air Flow(cfm)',]
original_2 = original_2[new_column_order]
original_2.to_csv("data/original-data-2-utf.csv", encoding="utf-8", index=False)
original_2 = pd.read_csv("data/original-data-2-utf.csv")

def clean_row_off_condition(df, name_of_row, func, values_to_drop):
    df_row = df[name_of_row]
    
    for i in range(len(df_row)):
        # Use positional indexing with .iloc
        if i > 0 and func(float(df_row.iloc[i])):
            values_to_drop.append(i)
            print(df_row.iloc[i])
        
    return values_to_drop

def clean_row_off_tolorence(df,name_of_row,tolerance,values_to_drop):
    df_row = df[name_of_row]
    
    for i in range(len(df_row)):
        if i > 0 and abs(float(df_row.iloc[i-1])-float(df_row.iloc[i])) >= tolerance:
            values_to_drop.append(i-2)
            values_to_drop.append(i-1)
            values_to_drop.append(i)
            values_to_drop.append(i+1)
            print(f"{float(df_row.iloc[i])} is i, {float(df_row.iloc[i-1])} is i - 1, diff is {abs(float(df_row.iloc[i-1])-float(df_row.iloc[i]))}, i: {i}")
    return values_to_drop

#Expects a dataframe and cleans it according to research specifications and returns the clean dataframe
def clean_data(df):
    df = df.replace('---', np.nan)
    df = df.replace('#VALUE!', np.nan)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    values_to_drop = []
    
    values_to_drop = clean_row_off_tolorence(df, "Water Flow Rate(gpm)", 0.50, values_to_drop)
    values_to_drop = clean_row_off_tolorence(df, "Air Flow(cfm)", 250, values_to_drop)
    values_to_drop = clean_row_off_condition(df, "Water Temp(F)",lambda x : x < 100, values_to_drop)
    
    values_to_drop = list(set(values_to_drop))
    values_to_drop = [i for i in values_to_drop if 0 <= i < len(df)]
    df = df.drop(values_to_drop)
    
    return df
    
clean_1 = clean_data(original_1)
clean_2 = clean_data(original_2)
clean = pd.concat([clean_1,clean_2])
clean.to_csv('data/clean-data-temp.csv', index=False)

# #Delete anything below 100T of in water, also remove delta of 0.5 gpm, remove deltas of 250cfm, remove 2 goofy points