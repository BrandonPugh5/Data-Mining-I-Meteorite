# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:33:37 2020

@author: Aaron Gunter
Edited by Aaron Gunter, Brandon Pugh
"""


import pandas as pd
import numpy as np
import pydotplus
from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.externals.six import StringIO
import os


#os.chdir(r'C:\Users\Aaron Gunter\Documents\Rowan\Data Mining\Final project')
pd.options.display.width = 0
meteoriteData = pd.read_csv("Meteorite_Landings.csv")

meteoriteData.head()

adjustedTData = pd.DataFrame(meteoriteData).copy()

adjustedTData.loc[adjustedTData['year'] <= 1750, 'year'] = '<1750'
adjustedTData.loc[(meteoriteData['year'] >= 1751) & (meteoriteData['year'] <= 1800), 'year'] = '1750-1800'
adjustedTData.loc[(meteoriteData['year'] >= 1801) & (meteoriteData['year'] <= 1850), 'year'] = '1801-1850'
adjustedTData.loc[(meteoriteData['year'] >= 1851) & (meteoriteData['year'] <= 1900), 'year'] = '1851-1900'
adjustedTData.loc[(meteoriteData['year'] >= 1901) & (meteoriteData['year'] <= 1950), 'year'] = '1901-1950'
adjustedTData.loc[(meteoriteData['year'] >= 1951) & (meteoriteData['year'] <= 2000), 'year'] = '1951-2000'
adjustedTData.loc[meteoriteData['year'] > 2000, 'year'] = '>2000'
adjustedTData['year'] =  adjustedTData['year'].astype(str)


adjustedTData["mass"].fillna(0, inplace = True)
adjustedTData['mass'] =  adjustedTData['mass'].astype(int)

adjustedTData.loc[meteoriteData['mass'] == 0, 'mass'] = 'No Record Found'
adjustedTData.loc[(meteoriteData['mass'] > 0) & (meteoriteData['mass'] <= 1), 'mass'] = '<1'
adjustedTData.loc[(meteoriteData['mass'] > 1) & (meteoriteData['mass'] <= 2), 'mass'] = '1-2'
adjustedTData.loc[(meteoriteData['mass'] > 2) & (meteoriteData['mass'] <= 3), 'mass'] = '2-3'
adjustedTData.loc[(meteoriteData['mass'] > 3) & (meteoriteData['mass'] <= 4), 'mass'] = '3-4'
adjustedTData.loc[(meteoriteData['mass'] > 4) & (meteoriteData['mass'] <= 5), 'mass'] = '4-5'
adjustedTData.loc[(meteoriteData['mass'] > 5) & (meteoriteData['mass'] <= 6), 'mass'] = '5-6'
adjustedTData.loc[(meteoriteData['mass'] > 6) & (meteoriteData['mass'] <= 10), 'mass'] = '6-10'
adjustedTData.loc[(meteoriteData['mass'] > 10) & (meteoriteData['mass'] <= 35), 'mass'] = '10-35'
adjustedTData.loc[(meteoriteData['mass'] > 35) & (meteoriteData['mass'] <= 50), 'mass'] = '35-50'
adjustedTData.loc[(meteoriteData['mass'] > 50) & (meteoriteData['mass'] <= 100), 'mass'] = '50-100'
adjustedTData.loc[(meteoriteData['mass'] > 100) & (meteoriteData['mass'] <= 300), 'mass'] = '100-300'
adjustedTData.loc[(meteoriteData['mass'] > 300) & (meteoriteData['mass'] <= 1000), 'mass'] = '300-1000'
adjustedTData.loc[meteoriteData['mass'] > 1000, 'mass'] = '>1000'

adjustedTData['mass'] =  adjustedTData['mass'].astype(str)

adjustedTData.loc[adjustedTData['country'] == '', 'Country'] = 'Location not found'

print(adjustedTData[['nametype', 'fall', 'recclass', 'mass', 'year', 'country']].head())


print(adjustedTData.columns)
print('\nName')
print(adjustedTData['nametype'].value_counts())
print('\nFell or Found')
print(adjustedTData['fall'].value_counts())
print('\nClass')
print(adjustedTData['recclass'].value_counts())
print('\nMass')
print(adjustedTData['mass'].value_counts())
print('\nYear')
print(adjustedTData['year'].value_counts())
print('\nCountry')
print(adjustedTData['country'].value_counts())

meteoriteDecTree = adjustedTData[['nametype', 'recclass', 'mass', 'fall', 'year', 'country']]
meteorite_with_dummies = pd.get_dummies(meteoriteDecTree)


#print(meteorite_with_dummies.head())
print(meteorite_with_dummies.shape)

colList = meteorite_with_dummies.columns
print(colList)

columnList3=['nametype_Relict', 'nametype_Valid', 'mass_0', 'mass_1-2',
       'mass_10-35', 'mass_100-300', 'mass_2-3', 'mass_3-4',
       'mass_300-1000', 'mass_35-50', 'mass_4-5', 'mass_5-6',
       'mass_50-100', 'mass_6-10', 'mass_<1', 'mass_>1000',
       'mass_No Record Found', 'year_1801-1850', 'year_1750-1800',
       'year_1851-1900', 'year_1901-1950', 'year_<1750',
       'year_>2000']

"""
file = open('columns.txt', 'w+')
try:
    for col_name in meteorite_with_dummies.columns:
        file.write(col_name + '\n')

except Exception as ex:
    print('encountered error')
    print(type(ex))
    print(ex)
finally:
    file.close()
"""

X=meteorite_with_dummies[columnList3]
Y=meteorite_with_dummies[['country_United_States', 'year_1951-2000']]



print(X[0:10])
print(Y[0:10])


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
c5 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
c5 = c5.fit(X_train,Y_train)
Y_pred = c5.predict(X_test)

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
export_graphviz(c5, out_file='dot_data',filled=True,rounded=True,special_characters=False,feature_names = columnList3)

with open("dot_data") as content_file:
    dot_data.write(content_file.read())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('meteoriteTree.png')


"""
Naive Bayes Classification
"""
nbC = BernoulliNB(alpha = 0)
nbC = nbC.fit(X, Y)
print(nbC.predict(X))
print(nbC.predict_proba(X))
print(nbC.score(X,Y))





