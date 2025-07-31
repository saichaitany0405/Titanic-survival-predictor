#importing all the necessary libraries
#All the commented "Prints" in this file are made so that i understand the data even better after each step!!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Loading dataset
df = pd.read_csv("train.csv")

#EDA - Data Inspection
# Trying to fetch the basic deatils from the train.csv file

# print("Shape of dataset:", df.shape)
# print(df.head())
# print(df.dtypes)
# print(df.isnull().sum())
# print(df['Survived'].value_counts())

#EDA - Data Cleaning
#dropping some unnecessary features,filling the null values and converting values for better understanding for the models.
df.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df['Sex']= df['Sex'].map({'male':0,'female':1})
df['Embarked']= df['Embarked'].map({'C':0,'Q':1,'S':2})
# print(df)

#EDA - Data Visualization(used to see realtions between prediction and Features)

#A plot on Total survivals and deaths
# sns.countplot(x='Survived', data=df)
# plt.title('Survival Distribution')
# plt.xlabel('Survived (0 = No, 1 = Yes)')
# plt.ylabel('Passenger Count')
# plt.show()

#Plotting relation between gender and Survival
# sns.countplot(x='Sex', hue='Survived', data=df)
# plt.title('Survival by Gender')
# plt.xlabel('Sex')
# plt.ylabel('Count')
# plt.legend(title='Survived', labels=['No', 'Yes'])
# plt.show()

#Plotting relation between Class(0 being lowest and 2 being highest class) and Survival
# sns.countplot(x='Pclass', hue='Survived', data=df)
# plt.title('Survival by Passenger Class')
# plt.xlabel('Passenger Class')
# plt.ylabel('Count')
# plt.legend(title='Survived', labels=['No', 'Yes'])
# plt.show()


# Setting x as features(every column except 'Survived') and y as prediction(Survived)
x = df.drop(['Survived'],axis=1)
y = df['Survived']
#using (x_train,x_test,y_train,y_test) as variables, same as i learnt in the machine learning class by Andrew NG
x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8,random_state=42) #80 percent of the data goes into traing and rest 20 percent is meant for testing; random_state ensures the same data is used for testing and training through out the model training
#choosing the LogisticRegression model as we are classifying data between Survived and Not Survived and fitting training data to LR model
model = LogisticRegression()
model.fit(x_train,y_train)
print('trained the model')

#Asking the LR model to predict from 20% features of x
y_pred = model.predict(x_test)
#Using diffrent metrics to see how well the model predicted
accuracy = accuracy_score(y_test,y_pred)
# print(accuracy)
# accuracy score = 0.8100558659217877

confu = confusion_matrix(y_test,y_pred)
# print(confu)
# confusion_matrix = [[90 15]
                    # [19 55]] (Gives True/Flase - Positives/Negatives)



