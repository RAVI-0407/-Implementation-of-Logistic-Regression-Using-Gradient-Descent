# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value

## Program and Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RAVIPRASATH K
RegisterNumber:  212224230225
*/
```
```
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv('drive/MyDrive/ML/Placement_Data.csv') 
data.head()
```
![image](https://github.com/user-attachments/assets/bf2337b2-4151-4244-964c-61b4688f6623)
```
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1
```
![image](https://github.com/user-attachments/assets/e9113f67-0db4-4753-bf56-e1cf13a13e8d)
```
data=data.drop('sl_no',axis=1) 
data=data.drop('salary',axis=1) 

data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes 
```
![image](https://github.com/user-attachments/assets/7ebe7046-e1ce-4e3f-b7cd-93c1a0892397)
```
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data 
```
![image](https://github.com/user-attachments/assets/69f0697c-e060-4f7d-ae94-2ff420a9e1ca)
```
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values

theta = np.random.randn(x.shape[1]) 
Y=y 

def sigmoid(z): 
   return 1/(1+np.exp(-z))
def loss(theta,X,y): 
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations): 
  m=len(y)
  for i in range(num_iterations): 
    h=sigmoid(X.dot(theta)) 
    gradient = X.T.dot(h-y)/m 
    theta-=alpha * gradient 
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta)) 
  y_pred=np.where(h>=0.5,1,0) 
  return y_pred 

y_pred = predict(theta,x) 
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
```
![image](https://github.com/user-attachments/assets/42299f02-580b-4a33-8ee2-deac9549a0e7)
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print("Predicted Result:",y_prednew)
```
![image](https://github.com/user-attachments/assets/d3c9be8e-c240-49b8-b3e2-a7eec682c456)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

