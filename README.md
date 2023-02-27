# Logistic_regression_model_SocialNetAds
build a model using Logistic Regression Clasifier and Make predictions , Finding Accuracy Score,vConfusion Matrix, Cross Validation Score, Classification Report, ROC-AUC Score
# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
# AOC ROC curve

from sklearn.metrics import roc_auc_score,roc_curve

# for Data Visualization
import matplotlib.pyplot as plt 
data=pd.read_csv('Social_Network_Ads.csv')
data.head()
# EDA --> Exploratory Data Analysis

data.shape
data.describe
data.info
# Encoding Categorical data into Numeric Data

le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])
data.head()
# For Training and Testing Data

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
print(x)
# Train and Test Data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=0)
print(x_train)
print(x_test)
# Scale Data using Standard Scalar

sc=StandardScaler()
sc.fit_transform(x_train,x_test)
# Model Creation Using Logistic Regression

lr=LogisticRegression()
# fit Training into Model
lr.fit(x_train,y_train)
# making Prediction

y_pred=lr.predict(x_test)
print(y_pred)
print(y_test)
# Find accuracy Score and Confusion Metrics

Accuracy_Score=accuracy_score(y_test,y_pred)
print(Accuracy_Score)
con_Mat=confusion_matrix(y_test,y_pred)
print(con_Mat)
cl_report=classification_report(y_test,y_pred)
print(cl_report)

# ROC Auc Score
roc_auc_score=roc_auc_score(y_test,y_pred)
print(roc_auc_score)


# Visualization 
x=x_test[:,0]
y=x_test[:,-1]
# For testing Data
c=y_test

plt.scatter(x,y,c=c)

# for prediction
c=y_pred

plt.scatter(x,y,c=c)
plt.title('on predict model')
plt.xlabel('x_test')
plt.ylabel('y_pred')

