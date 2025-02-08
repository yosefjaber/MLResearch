from sklearn.model_selection import train_test_split
import pandas as pd

#Read clean data
data = pd.read_csv("data/clean-data.csv")

#Create a X and y set
X = data.drop(columns=["AHU-01 VAV : Discharge Air Temperature (°F)"])
y = data["AHU-01 VAV : Discharge Air Temperature (°F)"]

#Create the test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)