# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1:Start

STEP 2:Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

STEP 3:Split the data into training and test sets using train_test_split.

STEP 4:Create and fit a logistic regression model to the training data.

STEP 5:Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

STEP 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

STEP 7:End

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ashwin Kumar A
RegisterNumber: 212223040021
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
dta
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

```

## Output:

![image](https://github.com/user-attachments/assets/70daec72-ba7c-452d-94e8-c3c8a2a7332e)
![image](https://github.com/user-attachments/assets/83e6aab0-377a-451c-8195-20b4ee73bb33)
![image](https://github.com/user-attachments/assets/80c14aca-7786-4ae6-8896-af16a0156c54)
![image](https://github.com/user-attachments/assets/d5d8cf55-5dec-414e-9290-1cba5ad36f8a)
![image](https://github.com/user-attachments/assets/73cd8dc6-f427-41b2-9b49-c614032e285b)
![image](https://github.com/user-attachments/assets/0ec9d756-5cfc-455b-b099-e7b82caa3011)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
