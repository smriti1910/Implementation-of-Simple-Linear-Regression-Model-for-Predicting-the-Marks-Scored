# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SMRITI .B
RegisterNumber: 212221040156
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
dataset.head()
dataset.tail()
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test, color="blue");

plt.plot(x_test, reg.predict(x_test), color="silver")

plt.title("Test set (H vs 5)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print('RMSE=',rmse)
b=np.array([[10]])
y_pred1=reg.predict(b)
print(y_pred1)

## Output:
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/79d63ea8-49cf-43ec-94be-91ae46cd8f56)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/6ae816f6-3f1b-499c-9551-ba7128669607)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/7c24ce83-ed20-4338-8c08-628f9cd1c750)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/593acac3-b19e-41cf-a0b9-f7293c243c98)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/0a9b0593-f2d4-404f-9d3a-ff345fb08e10)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/e3e8a2cc-1359-4aab-b40c-8f2db2af6803)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/9181f570-df63-4877-a9e1-fd7dbdbf7ea9)
![image](https://github.com/smriti1910/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/133334803/64689f74-130f-4f60-9a9a-fb9713761706)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
