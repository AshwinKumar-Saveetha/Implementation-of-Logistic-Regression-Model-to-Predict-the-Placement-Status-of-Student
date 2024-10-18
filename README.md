# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data**: Read the dataset.
2. **Preprocess**: Drop unnecessary columns and check for missing values.
3. **Encode**: Convert categorical features to numerical values using `LabelEncoder`.
4. **Split Data**: Divide into training and testing sets (80/20).
5. **Train Model**: Fit a `LogisticRegression` model on the training data.
6. **Predict**: Generate predictions for the test set.
7. **Evaluate**: Calculate accuracy, confusion matrix, and classification report.

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
data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)
data1.head()
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
from sklearn.metrics import classification_report 
classification_report1 = classification_report(y_test,y_pred) 
print(classification_report1)
```

## Output:

![image](https://github.com/user-attachments/assets/fead22be-241b-44f1-9452-dbef9e082cfb)

![image](https://github.com/user-attachments/assets/1b75be37-f56e-47ae-8fbe-e1bb3466927e)

![image](https://github.com/user-attachments/assets/a3768c13-8140-4991-b642-0305a4296298)

![image](https://github.com/user-attachments/assets/d5e3b26e-0998-4150-8942-5f29e8ab1b31)

![image](https://github.com/user-attachments/assets/e2307ff8-e131-41a3-ae29-f6c0ffd910ca)

![image](https://github.com/user-attachments/assets/a1416522-ed8c-433f-9e96-36cd47ce03a6)

![image](https://github.com/user-attachments/assets/2d78c3df-5fe5-4925-9b9e-c7b1377d68b0)

![image](https://github.com/user-attachments/assets/5ff45916-2eb3-4470-9298-5b693d482829)

![image](https://github.com/user-attachments/assets/4d010145-e387-4cc3-9d54-7da5d5af04af)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
