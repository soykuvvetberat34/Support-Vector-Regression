from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\SVR(Support Vector Regression)\\aylara_gore_satis.csv")
months=datas.iloc[:,0].values.reshape(-1,1)
sales=datas.iloc[:,1].values.reshape(-1,1)
sc=StandardScaler()
months_sc=sc.fit_transform(months)
sales_sc=sc.fit_transform(sales)

SVR_reg=SVR(kernel="rbf")
SVR_reg.fit(months_sc,sales_sc)
predict=SVR_reg.predict(months_sc)
plt.scatter(months_sc,sales_sc)
plt.plot(months_sc,predict)
plt.show()

















