# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 15:46:54 2017

@author: Charlotte
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Pretty display for notebooks

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
for col in data.columns:
    print '{}\t\t: count: {}, min: {:.0f}, max: {:.0f}, mean: {:.0f}, std: {:.0f}'.format(col, \
            data[col].count(), data[col].min(), data[col].max(),data[col].mean(),data[col].std()) 
    
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [44,134,333]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

scaler= preprocessing.MinMaxScaler()
scaledData = pd.DataFrame (scaler.fit_transform(data),columns=data.keys())
#scaledData['mean']=scaledData.mean(axis=1)

print scaledData.head()

i=0
colors=['red', 'blue', 'green', 'yellow', 'magenta', 'black', 'pink']
for col in scaledData.columns:
#    plt.scatter(list(range(len(scaledData[:,i]))),scaledData[:,i], color=colors[i], alpha=0.5 )
    plt.scatter(list(range(len(scaledData))),scaledData[col], color=colors[i], alpha=0.5,label=col )
    i+=1
#plt.legend()
plt.show()

#scaledData['mean']=scaledData.mean(axis=1)
barInput = pd.DataFrame(columns = scaledData.keys(), index=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
def below (col):
    result=[0]*10
    for i in col:
        for j in range(10):
            if j*0.1 < i and i <= (j+1)*0.1:
                result[j] +=1
    return result
    
for c in scaledData.keys():
    barInput[c]=below(scaledData[c])
print barInput

    
barInput = barInput.apply(lambda x: x/x.sum())
pd.options.display.float_format = '{:.0%}'.format
print barInput

#barInput.plot()
#plt.hist(barInput.index, barInput)

barInput.hist()
'''
ind = np.arange(len(barInput))    # the x locations for the groups
i=0
for bar in barInput:
    plt.bar(np.arange(len(bar))+i*0.1, bar, width=0.1, label=barInput.index[i], color=colors[i])
    i+=1
plt.legend()
plt.ylabel('number customer')
#plt.xticks(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
'''
plt.show()
