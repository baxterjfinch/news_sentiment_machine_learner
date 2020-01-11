import csv
import numpy as np
from numpy import loadtxt, set_printoptions
from pandas import read_csv, set_option
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Custom Functions
from feature_selection_techniques import univariate_selection, recursive_feature_elimination, principal_component_analysis, feature_importance
from class_distribution import get_class_distribution

### Classifications

def getCSV(path):
    return r"{}".format(path)

def getFileData(opened_file):
    with open(opened_file, 'r') as f:
        reader = csv.reader(f, delimiter= ',')
        headers = next(reader)
        data = list(reader)
        data = np.array(data)

        return {
            "headers": headers,
            "body": data
        }

def getPandaParsed(path, names=None):
    return read_csv(path, names=names)

def preprocessDataScaler(array):
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    return data_scaler.fit_transform(array)



file = getCSV("iris.csv")
# data = getFileData(file)

## Panda Stuff

###### Pima Indians Diabetes Data ######
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = getPandaParsed("pima.csv", names)

####### DATA PRE-PROCESSING/AGGREGATION ########
## Gets first 50 rows
# print(data.head(5))

## Gets size of data
# print(data.shape)

## Gets data types
# print(data.dtypes)

# set_option('display.width', 100)
# set_option('precision', 2)

## Gets columns Count, Mean, STD, Min, 25%, 50%, 75%, Max
# print(data.describe())


### Class Distribution
# data = getPandaParsed("iris.csv")
# count_class = get_class_distribution(data, 'sepal.length')
# print(count_class)

## Correlation Between Attributes
# set_option('display.width', 100)
# set_option('precision', 2)
# correlations = data.corr(method='pearson')
# print(correlations)

## Calculate Skew
# print(data.skew())

### Scale Data
# array = data.values
# data_rescaled = preprocessDataScaler(array)
# print(data_rescaled)


######## FEATURE SELECTION TECHNIQUES ##########

### Univariate Selection
## Selects features with the help of stat testing and prediction variable relationships

array = dataframe.values

# Separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

### Summarize the data for output: Sets precision to 2 and shows 4 top feature attribtues w/ scores
# featured_data = univariate_selection(X, Y, 4, 2)
# print ("\nFeatured data:\n", featured_data[0:4])

### Select best 3 attributes using LogisticRegression algorithm
### NOT WORKING GREAT
# fit = recursive_feature_elimination(array, X, Y, 3)
# print(fit)


### Principle Component Analysis (linear algebra) transforms dataset into compressed form
# fit = principal_component_analysis(array, X, Y, 3)


### Feature Imporance
# model = feature_importance(array, X, Y)



########## Classifications ##########
