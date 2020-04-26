# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:33:37 2020

@author: Aaron Gunter
"""


import pandas as pd
import pydotplus
from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.externals.six import StringIO
import os


os.chdir(r'C:\Users\Aaron Gunter\Documents\Rowan\Data Mining\HW 7\titanic')
pd.options.display.width = 0
titanicData = pd.read_csv("train.csv")

titanicData.head()

adjustedTData = pd.DataFrame(titanicData).copy()

adjustedTData.loc[adjustedTData['Survived'] == 0, 'Survived'] = 'No'
adjustedTData.loc[adjustedTData['Survived'] == 1, 'Survived'] = 'Yes'
adjustedTData.loc[adjustedTData['Pclass'] == 1, 'Pclass'] = 'Upper'
adjustedTData.loc[adjustedTData['Pclass'] == 2, 'Pclass'] = 'Middle'
adjustedTData.loc[adjustedTData['Pclass'] == 3, 'Pclass'] = 'Lower'
adjustedTData.loc[adjustedTData['Age'] <= 5, 'Age'] = '0-5'
adjustedTData.loc[(titanicData['Age'] >= 6) & (titanicData['Age'] <= 10), 'Age'] = '6-10'
adjustedTData.loc[(titanicData['Age'] >= 11) & (titanicData['Age'] <= 15), 'Age'] = '11-15'
adjustedTData.loc[(titanicData['Age'] >= 16) & (titanicData['Age'] <= 20), 'Age'] = '16-20'
adjustedTData.loc[(titanicData['Age'] >= 21) & (titanicData['Age'] <= 30), 'Age'] = '21-30'
adjustedTData.loc[(titanicData['Age'] >= 31) & (titanicData['Age'] <= 40), 'Age'] = '31-40'
adjustedTData.loc[(titanicData['Age'] >= 41) & (titanicData['Age'] <= 50), 'Age'] = '41-50'
adjustedTData.loc[(titanicData['Age'] >= 51) & (titanicData['Age'] <= 60), 'Age'] = '51-60'
adjustedTData.loc[(titanicData['Age'] >= 61) & (titanicData['Age'] <= 70), 'Age'] = '61-70'
adjustedTData.loc[(titanicData['Age'] >= 71) & (titanicData['Age'] <= 80), 'Age'] = '71-80'
adjustedTData.loc[titanicData['Age'] > 80, 'Age'] = '>80'
adjustedTData['Age'] =  adjustedTData['Age'].astype(str)
adjustedTData.loc[adjustedTData['Age'] == 'nan', 'Age'] = 'Unknown'

adjustedTData.loc[adjustedTData['Fare'] <= 10.0, 'Fare'] = '0-10'
adjustedTData.loc[(titanicData['Fare'] > 10.0) & (titanicData['Fare'] <= 20.0), 'Fare'] = '>10-20'
adjustedTData.loc[(titanicData['Fare'] > 20.0) & (titanicData['Fare'] <= 30.0), 'Fare'] = '>20-30'
adjustedTData.loc[(titanicData['Fare'] > 30.0) & (titanicData['Fare'] <= 40.0), 'Fare'] = '>30-40'
adjustedTData.loc[(titanicData['Fare'] > 40.0) & (titanicData['Fare'] <= 50.0), 'Fare'] = '>40-50'
adjustedTData.loc[(titanicData['Fare'] > 50.0) & (titanicData['Fare'] <= 60.0), 'Fare'] = '>50-60'
adjustedTData.loc[(titanicData['Fare'] > 60.0) & (titanicData['Fare'] <= 70.0), 'Fare'] = '>60-70'
adjustedTData.loc[(titanicData['Fare'] > 70.0) & (titanicData['Fare'] <= 80.0), 'Fare'] = '>70-80'
adjustedTData.loc[(titanicData['Fare'] > 80.0) & (titanicData['Fare'] <= 90.0), 'Fare'] = '>80-90'
adjustedTData.loc[(titanicData['Fare'] > 90.0) & (titanicData['Fare'] <= 100.0), 'Fare'] = '>90-100'
adjustedTData.loc[titanicData['Fare'] > 100.0, 'Fare'] = '>100'
adjustedTData['Fare'] =  adjustedTData['Fare'].astype(str)
adjustedTData.loc[adjustedTData['Fare'] == 'nan', 'Fare'] = 'Unknown'

print(adjustedTData[['Survived', 'Pclass', 'Age', 'Fare']].head())

print(adjustedTData.columns)
print('\nSurvived')
print(adjustedTData['Survived'].value_counts())
print('\nPclass')
print(adjustedTData['Pclass'].value_counts())
print('\nName')
print(adjustedTData['Name'].value_counts())
print('\nSex')
print(adjustedTData['Sex'].value_counts())
print('\nAge')
print(adjustedTData['Age'].value_counts())
print('\nSibSp')
print(adjustedTData['SibSp'].value_counts())
print('\nParch')
print(adjustedTData['Parch'].value_counts())
print('\nTicket')
print(adjustedTData['Ticket'].value_counts())
print('\nFare')
print(adjustedTData['Fare'].value_counts())
print('\nCabin')
print(adjustedTData['Cabin'].value_counts())
print('\nEmbarked')
print(adjustedTData['Embarked'].value_counts())



titanicDecTree = adjustedTData[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
titanic_with_dummies = pd.get_dummies(titanicDecTree)

print(titanic_with_dummies.head())
print(titanic_with_dummies.shape)

colList = titanic_with_dummies.columns
print(colList)

columnList2=[u'SibSp', u'Parch', u'Pclass_Lower',
       u'Pclass_Middle', u'Pclass_Upper', u'Sex_male', u'Age_0-5',
       u'Age_11-15', u'Age_16-20', u'Age_20.5', u'Age_21-30', u'Age_30.5',
       u'Age_31-40', u'Age_40.5', u'Age_41-50', u'Age_51-60', u'Age_6-10',
       u'Age_61-70', u'Age_70.5', u'Age_71-80', u'Age_Unknown', u'Fare_0-10',
       u'Fare_>10-20', u'Fare_>100', u'Fare_>20-30', u'Fare_>30-40', u'Fare_>40-50',
       u'Fare_>50-60', u'Fare_>60-70', u'Fare_>70-80', u'Fare_>80-90',
       u'Fare_>90-100', u'Embarked_C', u'Embarked_Q', u'Embarked_S', u'Survived_Yes']

columnList3=[u'SibSp', u'Parch', u'Pclass_Lower',
       u'Pclass_Middle', u'Pclass_Upper', u'Sex_male', u'Age_0-5',
       u'Age_11-15', u'Age_16-20', u'Age_20.5', u'Age_21-30', u'Age_30.5',
       u'Age_31-40', u'Age_40.5', u'Age_41-50', u'Age_51-60', u'Age_6-10',
       u'Age_61-70', u'Age_70.5', u'Age_71-80', u'Age_Unknown', u'Fare_0-10',
       u'Fare_>10-20', u'Fare_>100', u'Fare_>20-30', u'Fare_>30-40', u'Fare_>40-50',
       u'Fare_>50-60', u'Fare_>60-70', u'Fare_>70-80', u'Fare_>80-90',
       u'Fare_>90-100', u'Embarked_C', u'Embarked_Q', u'Embarked_S']


X=titanic_with_dummies[columnList3]
Y=titanic_with_dummies.Survived_Yes
"""
print(X[0:10])
print(Y[0:10])
"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
c5 = DecisionTreeClassifier(criterion='entropy', max_depth=4)
c5 = c5.fit(X_train,Y_train)
Y_pred = c5.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
export_graphviz(c5, out_file='dot_data',filled=True,rounded=True,special_characters=False,feature_names = columnList3)

with open("dot_data") as content_file:
    dot_data.write(content_file.read())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanicTree.png')



"""
Naive Bayes Classification
"""
nbC = BernoulliNB(alpha = 0)
nbC = nbC.fit(X, Y)
print(nbC.predict(X))
print(nbC.predict_proba(X))
print(nbC.score(X,Y))





