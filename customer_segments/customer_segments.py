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
plt.legend(bbox_to_anchor=(1.3, 1.0))
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

    
barInput = barInput.apply(lambda x: x/x.sum())
pd.options.display.float_format = '{:.0%}'.format
print barInput

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

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

for col in data.columns:
    X = pd.DataFrame(data,copy=True)
    y = X[col]
    X.drop([col], axis = 1, inplace = True)
    #print col, '\n',X.head(1)

    # TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state=33, max_depth=5)
    regressor.fit(X_train,y_train)


    # TODO: Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    
    print 'column:', col, ' score:', score
    
# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# For each feature find the data points with extreme high or low values
pd.options.display.float_format = '{:,.2f}'.format
outliers  = []
outliers2  = {}

for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*1.5
    
    # Display the outliers
    print feature, ': Q1-step:', Q1-step, ' Q3:', Q3+step
    print "Data points considered outliers for the feature '{}':".format(feature)
    helper=log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    for i in helper.index:
        if i in outliers2:
            outliers2[i].append(feature)
        else:
            outliers2[i]=[feature]
    display(helper)
    
for key,value in outliers2.iteritems():
    if len(value)> 1:
        print key,'\t:',value
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = outliers2.keys()
print outliers

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
pd.scatter_matrix(good_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

print 'found {} outliers over all features'.format(len(outliers))


# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# TODO: Apply PCA by fitting the good data with only two dimensions
pca =  PCA(n_components=2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])