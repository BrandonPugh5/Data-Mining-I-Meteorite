# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:22:04 2020

@author: Aaron Gunter
"""

import pandas as pd
import numpy as np
import os
from collections import Counter


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers



os.chdir(r'C:\Users\Aaron Gunter\Documents\Rowan\Data Mining\Project')
meteoriteLandings = pd.read_csv("Meteorite_Landings.csv")

meteoriteLandings.head()


Outliers_to_drop = detect_outliers(meteoriteLandings,2,["mass (g)","SibSp","Parch","Fare"])


"""
Yamato meteorites likely fell somewhere else and were moved to a "collection point" over time on ice flows.
"""


meteoriteLandings = meteoriteLandings.fillna(np.nan)

meteoriteLandings.isnull().sum()


meteoriteLandings.info()
meteoriteLandings.dtypes

meteoriteLandings.describe()