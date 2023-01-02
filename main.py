import pandas as pd
import numpy as np
import copy
import joblib

import tensorflow as tf
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt

dataset_cols = ["Car_Name","Year","Selling_Price","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"]
df = pd.read_csv("car_data.csv")

print(df)



for label in df.columns[1:]:
  plt.scatter(df[label], df["Selling_Price"])
  plt.title(label)
  plt.ylabel("Selling_Price")
  plt.xlabel(label)
  plt.show()


train, val, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

def get_xy(dataframe, y_label, x_labels=None):
  dataframe = copy.deepcopy(dataframe)
  if x_labels is None:
    X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
  else:
    if len(x_labels) == 1:
      X = dataframe[x_labels[0]].values.reshape(-1, 1)
    else:
      X = dataframe[x_labels].values

  y = dataframe[y_label].values.reshape(-1, 1)
  data = np.hstack((X, y))

  return data, X, y

_, X_train_temp, y_train_temp = get_xy(train, "Selling_Price", x_labels=["Present_Price"])
_, X_val_temp, y_val_temp = get_xy(val, "Selling_Price", x_labels=["Present_Price"])
_, X_test_temp, y_test_temp = get_xy(test, "Selling_Price", x_labels=["Present_Price"])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
print("Coeficient " + str(temp_reg.coef_))
print("Intercept" + str(temp_reg.intercept_))

LinearRegression()
print("r linear regression  trained : ")
print(temp_reg.score(X_test_temp, y_test_temp))

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(temp_reg, filename)
