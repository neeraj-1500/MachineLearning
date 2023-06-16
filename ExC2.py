from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
df = pd.read_csv(r"C:\Users\Electrobot\PycharmProjects\MachineLearning\MachineLearning\datasets\housing\housing.csv")
print("df.head----->",df.head())
print(df.describe())
print(df.info)
#Steps
#find useful attributes to the target variable i.e finding the correlated attributes
df.hist(bins=50, figsize=(10,20))
"""We are doing this to see the spread of data across various entries, the bins parameter

Simple answer: bins should be the number of bars you want to show in your histogram plot.
But let's unwrap the chain: Pandas hist function calls matplotlib's hist function. In contrast to pandas, matplotlib has a verbose docstring,
bins : integer or sequence or ‘auto’, optional
If an integer is given, bins + 1 bin edges are calculated and returned, consistent with numpy.histogram().
If bins is a sequence, gives bin edges, including left edge of first bin and right edge of last bin. In this case, bins is returned unmodified.
All but the last (righthand-most) bin is half-open. In other words, if bins is:
[1, 2, 3, 4] then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4.
Unequally spaced bins are supported if bins is a sequence.
It should be noted that by default, the numpy default value of 10 bins between the minimum and maximum datapoint are chosen.
This means that the data range is divided into 10 equally sized intervals and any value is assigned to one of those 10 bins,
adding up to the value of the bin. This value will then be shown as the height of the respective bar in the plot.

The native figure size unit in Matplotlib is inches, deriving from print industry standards. However, 
users may need to specify their figures in other units like centimeters or pixels. This example illustrates how to do this efficiently.
plt.subplots(figsize=(15*cm, 5*cm))

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.subplots(figsize=(600*px, 200*px))
 """
plt.show()
df_num=df.drop(columns=["ocean_proximity"],inplace=False, axis=1)
print(df_num.columns)
"""
We can do a StratifiedShuffleSplit on a attribute that is highly 
correlated with the target variable
StratifiedShuffleSplit steps:
find intervals in which data is largely populated,
you can use histogram to check the data spread.
split the data using pd.cut into groups and label them 
according to the interval we can labels 1,2,3 etc because we
will not be using this attribute for training our model as it 
will be used to split the data into train_set and test_set
such that all entries of highly correlated attributes  are uniformly
distrubted across train_set and test_set
"""
corr_matrix=df_num.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=True))
scaled_median_income = pd.cut(df["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
scaled_housing_median_age = pd.cut(df["housing_median_age"], bins=[0,10,20,30,40,np.inf],labels=[1,2,3,4,5])
scaled_median_income.hist()
plt.show()
scaled_housing_median_age.hist()
plt.show()
print(df.shape, scaled_median_income.shape, scaled_housing_median_age.shape)
df["income_cat"]= scaled_median_income
df["housing_age_cat"]=scaled_housing_median_age
print(df.head())

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(df, df["housing_age_cat"]):
    train_set = df.iloc[train_index]
    test_set =  df.iloc[test_index]

print("len(train_set)-->"+str(len(train_set)))
print("len(test_set)-->"+str(len(test_set)))
print(df["housing_age_cat"].value_counts()/len(df))
print(train_set["housing_age_cat"].value_counts()/len(train_set))

#Drop the attributes used for splitting as it will not be passed as input
# to our machine learning algorithm

for set_ in (train_set,test_set):
    set_.drop(columns=["income_cat","housing_age_cat"], inplace=True, axis=1)

print(train_set.columns)
print(test_set.columns)

train_set.plot(kind="scatter",x="longitude",y="latitude", alpha=0.1)
plt.show()

train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=train_set["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()
plt.show()

"""Experimenting with attributes"""
train_set["rooms_per_household"]=train_set["total_rooms"]/train_set["population"]
test_set["rooms_per_household"]=train_set["total_rooms"]/test_set["population"]
train_set["bedrooms_per_household"]=train_set["total_bedrooms"]/train_set["population"]
test_set["bedrooms_per_household"]=test_set["total_bedrooms"]/test_set["population"]
train_corr_set = train_set.drop(columns=["ocean_proximity"],axis=1)
train_corr = train_corr_set.corr()["median_house_value"]
print(train_corr)





