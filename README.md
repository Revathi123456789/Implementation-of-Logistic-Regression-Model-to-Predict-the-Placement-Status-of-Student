# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: REVATHI
RegisterNumber:  212223040169

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("/content/Placement_Data.csv")

print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print(data1.isnull().sum())

print(data1.duplicated().sum())

le = LabelEncoder()
categorical_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in categorical_cols:
    data1[col] = le.fit_transform(data1[col])

X = data1.iloc[:, :-1]
y = data1["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)

sample_input = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
sample_prediction = lr.predict(sample_input)
print("Sample Prediction:", sample_prediction)

*/
```

## Output:

![Screenshot 2025-04-03 230607](https://github.com/user-attachments/assets/57b1f6fd-f732-46af-b329-383c97df90b9)

![image](https://github.com/user-attachments/assets/cabe068b-d0a9-457e-9f97-273485c92198)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
