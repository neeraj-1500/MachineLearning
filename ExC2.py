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
train_housing = train_set.drop(columns="median_house_value", axis=1)
train_set_label = train_set[["median_house_value"]]
test_housing = test_set.drop(columns="median_house_value", axis=1)
test_set_label = test_set[["median_house_value"]]

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
train_set_copy = train_set.copy()
test_set_copy = test_set.copy()

"""Experimenting with attributes"""
train_set_copy["rooms_per_household"]=train_set_copy["total_rooms"]/train_set_copy["population"]
test_set_copy["rooms_per_household"]=test_set_copy["total_rooms"]/test_set_copy["population"]
train_set_copy["bedrooms_per_household"]=train_set["total_bedrooms"]/train_set["population"]
test_set_copy["bedrooms_per_household"]=test_set["total_bedrooms"]/test_set["population"]
train_set_copy_num = train_set_copy.drop(columns=["ocean_proximity"],axis=1)
train_corr = train_set_copy_num.corr()
print(train_corr)
print(train_corr["median_house_value"].sort_values(ascending=False))
print(type(train_corr))
""" Handling the missing data through SimpleImputer library, you can 
explore the startegy option"
strategystr, default=’mean’
The imputation strategy.
If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
"""

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
"""Here train_set_copy_num contains all the numerical attributes because median 
is not for categorical or textual value"""
imputer.fit(train_set_copy_num)
print(imputer.statistics_)
X=imputer.transform(train_set_copy_num)
housing_tr = pd.DataFrame(X, columns=train_set_copy_num.columns)

#Handling Text and Categorical Attributes
train_cat = train_set[["ocean_proximity"]]
print(train_cat.head())

""" Using the OneHotEncoder to handle the categorical attributes"""

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(train_cat)
print(housing_cat_1hot)
""" housing_cat_1hot is a sparse matrix when there are large number of categories
 the sparse matrix is handy. sparse matrix only stores the indexes of non zero values as
  dense matrix stores both o's and 1 it consumes more memory
If you want to convert the matrix to dense matrix use matrix.toarray() functions
  """

print(housing_cat_1hot.toarray())

"""Writing Custom Transformers--------"""
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
"""
You will want your transformer to work seamlessly with Scikit-Learn func‐
tionalities (such as pipelines), and since Scikit-Learn relies on duck typing (not inher‐
itance), all you need is to create a class and implement three methods: fit()
(returning self), transform(), and fit_transform(). You can get the last one for
free by simply adding TransformerMixin as a base class. Also, if you add BaseEstima
tor as a base class (and avoid *args and **kargs in your constructor) you will get
two extra methods (get_params() and set_params()) that will be useful for auto‐
matic hyperparameter tuning. For example, here is a small transformer class that adds
the combined attributes we discussed earlier
In Python, np.c_ is a short technique for concatenation of arrays using numpy1. 
It is used to concatenate arrays along the second axis12. It is short-hand for 
np.r_ ['-1,2,0', index expression]2. numpy.c_ returns a concatenated array3.
"""
class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:, population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder=CustomTransformer(add_bedrooms_per_room=False)
attr_adder.fit(housing_tr.values)
housing_tr_extra_attribs = attr_adder.transform(housing_tr.values)

"""Feature Scaling
There are two common ways to get all attributes to have the same scale: min-max
scaling and standardization
Min-max scaling (many people call this normalization) is quite simple: values are
shifted and rescaled so that they end up ranging from 0 to 1. We do this by subtract‐
ing the min value and dividing by the max minus the min. Scikit-Learn provides a
transformer called MinMaxScaler for this
min_max_scaler=(value-min_value)/(max_value-min_value)

Standardization is quite different: first it subtracts the mean value (so standardized
values always have a zero mean), and then it divides by the standard deviation so that
the resulting distribution has unit variance. Unlike min-max scaling, standardization
does not bound values to a specific range, which may be a problem for some algo‐
rithms (e.g., neural networks often expect an input value ranging from 0 to 1). How‐
ever, standardization is much less affected by outliers.
standardization = (value-mean)/standard_deviation
Scikit-Learn provides a transformer called StandardScaler for stand‐
ardization.
"""
#Transformation Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline ([
    ("imputer", SimpleImputer(strategy="median")),
    ('attribs_adder', CustomTransformer()),
        ("std_scaler", StandardScaler())
]
)
train_set_num = train_set.drop(columns=["ocean_proximity", "median_house_value"], axis=1)
train_set_cat = train_set[["ocean_proximity"]]

housing_tr_num = num_pipeline.fit_transform(train_set_num)
print("train_set.shape is -->", train_set_num.shape)
print("housing_tr_num.shape is -->",housing_tr_num.shape)


from sklearn.compose import ColumnTransformer
"""The constructor requires a list of tuples, where each
tuple contains a name22, a transformer and a list of names (or indices) of columns
that the transformer should be applied to. """
num_attribs = list(train_set_num.columns)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)

])
housing_train_data = train_set.drop(columns="median_house_value", axis=1)
housing_train_label = train_set[["median_house_value"]]
housing_test_data = test_set.drop(columns="median_house_value", axis=1)
housing_test_label = test_set[["median_house_value"]]


train_housing = full_pipeline.fit_transform(housing_train_data)
test_housing = full_pipeline.transform(housing_test_data)
print("------------ At line number 239-------------")
print("housing_train_data.shape is {} and housing_test_data.shape is {}".format(train_housing.shape, test_housing.shape))
print("housing_train_data.columns is {} and housing_test_data.columns is {}".format(housing_train_data.columns, housing_test_data.columns))

print("type of train_housing is -->"+str(type(train_housing)))
print("type of test_housing is -->"+str(type(test_housing)))

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_housing,train_set_label)

from sklearn.metrics import mean_squared_error
y_predict = lin_reg.predict(test_housing)
y_actual = test_set_label.values

print("y_predict is -->", type(y_predict))
print("y_actual is -->", type(y_actual))
print(y_predict.shape)
print(y_actual.shape)

print("y_predict is -->", y_predict[:10].tolist())
print("y_actual is -->", y_actual[:10].tolist())

lin_mse = mean_squared_error(y_actual,y_predict)
lin_rmse = np.sqrt(lin_mse)
print("Root mean squared error is ",lin_rmse)



from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_housing,train_set_label)
tree_reg_predictions = tree_reg.predict(test_housing)

tree_mse = mean_squared_error(y_actual, tree_reg_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse is -->"+str(tree_rmse))

#Better Evaluation Using Cross-Validation
def display_scores(scores):
    print("Scores:",scores)
    print("Mean:", scores.mean())
    print("Standard deviation", scores.std())



from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, train_housing, train_set_label, scoring="neg_mean_squared_error", cv=15)
tree_rmse_scores = np.sqrt(-scores)
print ("Scores for Decision Tree Regressor")
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, train_housing, train_set_label, scoring="neg_mean_squared_error", cv=15)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Scores for linear regression")
display_scores(lin_rmse_scores)

print("Mean of the error is less in case of"
       " linear_regression as compared to DecisionTree regressor hence we will go with linear regression")

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_housing, train_set_label.values.ravel())
forest_reg_score = cross_val_score(forest_reg,train_housing,train_set_label.values.ravel(),
                                   scoring="neg_mean_squared_error", cv =15)


forest_reg_rmse = np.sqrt(-forest_reg_score)
display_scores(forest_reg_rmse)


#Fine- Tune your model

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(train_housing, train_set_label.values.ravel())
"""
This param_grid tells Scikit-Learn to first evaluate all 3 × 4 = 12 combinations of
n_estimators and max_features hyperparameter values specified in the first dict
(don’t worry about what these hyperparameters mean for now; they will be explained
in Chapter 7), then try all 2 × 3 = 6 combinations of hyperparameter values in the
second dict, but this time with the bootstrap hyperparameter set to False instead of
True (which is the default value for this hyperparameter).
All in all, the grid search will explore 12 + 6 = 18 combinations of RandomForestRe
gressor hyperparameter values, and it will train each model five times (since we are
using five-fold cross validation). In other words, all in all, there will be 18 × 5 = 90
rounds of training! It may take quite a long time, but when it is done you can get the
best combination of parameters like this:
"""

print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.cv_results_)
breakpoint()
grid_search_results = grid_search.cv_results_

for mean_score, params in zip(grid_search_results["mean_test_score"],grid_search_results["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs =["rooms_per_hhlod","population_per_hhlod","bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
features = num_attribs+extra_attribs+cat_one_hot_attribs
print(sorted(zip(feature_importances, features),reverse=True) )

#Evaluating our system on the test set
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(test_housing)
mse = mean_squared_error(y_actual, final_predictions)
rmse = np.sqrt(mse)
print("final_rmse is -->", rmse)































