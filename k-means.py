import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report

meteor = pd.read_csv('Meteorite_Landings.csv')
meteor2 = pd.DataFrame(meteor).copy()

meteor2["mass"].fillna(0, inplace = True)
meteor2['mass'] =  meteor2['mass'].astype(int)

meteor2.loc[meteor['mass'] == 0, 'mass'] = 'No Record Found'
meteor2.loc[(meteor['mass'] > 0) & (meteor['mass'] <= 1), 'mass'] = '<1'
meteor2.loc[(meteor['mass'] > 1) & (meteor['mass'] <= 2), 'mass'] = '1-2'
meteor2.loc[(meteor['mass'] > 2) & (meteor['mass'] <= 3), 'mass'] = '2-3'
meteor2.loc[(meteor['mass'] > 3) & (meteor['mass'] <= 4), 'mass'] = '3-4'
meteor2.loc[(meteor['mass'] > 4) & (meteor['mass'] <= 5), 'mass'] = '4-5'
meteor2.loc[(meteor['mass'] > 5) & (meteor['mass'] <= 6), 'mass'] = '5-6'
meteor2.loc[(meteor['mass'] > 6) & (meteor['mass'] <= 10), 'mass'] = '6-10'
meteor2.loc[(meteor['mass'] > 10) & (meteor['mass'] <= 35), 'mass'] = '10-35'
meteor2.loc[(meteor['mass'] > 35) & (meteor['mass'] <= 50), 'mass'] = '35-50'
meteor2.loc[(meteor['mass'] > 50) & (meteor['mass'] <= 100), 'mass'] = '50-100'
meteor2.loc[(meteor['mass'] > 100) & (meteor['mass'] <= 300), 'mass'] = '100-300'
meteor2.loc[(meteor['mass'] > 300) & (meteor['mass'] <= 1000), 'mass'] = '300-1000'
meteor2.loc[meteor['mass'] > 1000, 'mass'] = '<1000'

meteor2['country'] =  meteor2['country'].astype(str)
meteor2['mass'] =  meteor2['mass'].astype(str)
plt.scatter(meteor2['year'], meteor2['country'])
#plt.xlim(0, 50000)
plt.xlim(1700, 2020)
plt.show()

meteor["mass"].fillna(0, inplace = True)
km = KMeans(n_clusters=3)

km_predicted = km.fit_predict(meteor[['mass', 'year']])

meteor['clutter'] = km_predicted
df1 = meteor[meteor.cluster==0]
df2 = meteor[meteor.cluster==1]
df3 = meteor[meteor.cluster==2]

plt.scatter(df1.country, df1['mass'], color='green')
plt.scatter(df2.country, df2['mass'], color='red')
plt.scatter(df3.country, df3['mass'], color='black')

plt.xlabel('Country')
plt.ylabel('mass')
plt.legend()

#plt.show()

#print(km_predicted)

"""Source Code
iris = datasets.load_iris()
X = scale(iris.data)
Y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
print(X[0:10,])

clustering = KMeans(n_clusters = 3, random_state = 5)
print(clustering.fit(X))

iris_df = pd.DataFrame(iris.data)

iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_length', 'Petal_Width']
Y.columns = ['Targets']

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.subplot(1, 2, 1)
plt.scatter(x=iris_df.Petal_length, y= iris_df.Petal_Width, c=color_theme[iris.target], s=50) 
plt.title('Ground Truth Classification')
plt.show()

plt.subplot(1, 2, 2)
plt.scatter(x=iris_df.Petal_length, y= iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title('K-Means Classification')
plt.show()

relabel = np.choose(clustering.labels_,[2, 0, 1]).astype(np.int64)
plt.subplot(1, 2, 1)
plt.scatter(x=iris_df.Petal_length, y= iris_df.Petal_Width, c=color_theme[iris.target], s=50) 
plt.title('Ground Truth Classification')
plt.show()

plt.subplot(1, 2, 2)
plt.scatter(x=iris_df.Petal_length, y= iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title('K-Means Classification')
plt.show()

print(classification_report(Y, relabel))
"""