import pandas as pd
import numpy as np
import copy
import tensorflow as tf
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt

dataset_cols = ["Car_Name","Year","Selling_Price","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"]
df = pd.read_csv("car_data.csv").drop(["Transmission", "Car_Name","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"], axis=1)

df.head()



for label in df.columns[1:]:
  plt.scatter(df[label], df["Year"])
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

_, X_train_temp, y_train_temp = get_xy(train, "Selling_Price", x_labels=["Year"])
_, X_val_temp, y_val_temp = get_xy(val, "Selling_Price", x_labels=["Year"])
_, X_test_temp, y_test_temp = get_xy(test, "Selling_Price", x_labels=["Year"])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
LinearRegression()
print("r linear regression  trained : ")
print(temp_reg.score(X_test_temp, y_test_temp))

plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Selling_Price vs Year")
plt.ylabel("Selling Price")
plt.xlabel("Year")
plt.show()

#Multiple linear regressions

print(df.head())

train, val, test = np.split(df.sample(frac=1), [int(0.2*len(df)), int(0.5*len(df))])
_, X_train_all, y_train_all = get_xy(train, "Selling_Price", x_labels=df.columns[0:])
_, X_val_all, y_val_all = get_xy(val, "Selling_Price", x_labels=df.columns[0:])
_, X_test_all, y_test_all = get_xy(test, "Selling_Price", x_labels=df.columns[0:])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)
print("r Linear regression Predicted: ")
print(all_reg.score(X_test_all, y_test_all))

y_pred_lr = all_reg.predict(X_test_all)


