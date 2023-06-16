import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path= HOUSING_PATH):
    # fetch_housing_data()
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())
housing.hist(bins=50, figsize=(20,15))
# plt.show()

def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    print(shuffled_indices)
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    print(test_indices)
    print(train_indices)
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6, np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    print("train_index is -->", train_index)
    print("test_index is -->", test_index)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat",inplace=True, axis=1)

print(type(strat_train_set))
print(type(strat_test_set))
print(dir(strat_test_set))
print(strat_test_set.shape)
print(strat_test_set.columns)
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
# plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100,
             label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"),
             colorbar="True")
plt.legend()
# plt.show()
#looking for correlations
ocean_proximity = housing["ocean_proximity"]
housing.drop("ocean_proximity",axis=1, inplace=True)
corr_matrix = housing.corr()
print(corr_matrix)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#Using scatter_matrix function to visualize the correlation between
#important attributes

from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
# plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

#Experimenting with Attribute Combinations

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]= housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#Prepare the data for machine learning algorithms

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#handling the missing values
median =housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)

#Handling missing data with SimpleImputer

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

#The imputer has simply computed the median of each attribute and store the result in its statistics_
print(imputer.statistics_)

#Now we can use this trained imputer to transform the training set by replacing the missing values

X = imputer.transform(housing_num)
#The result is a plain NumPy array containing the transformed features
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)

print(housing_extra_attribs)

# Min_max scaling is x-min/(Max-Min)
# Standardization is x-mean/(Standard Deviation) it has unit variance
#Min_max scaling sets the value in a range whereas standardization is much less effected by outliers

#Transformation Pipelines
#It is helpful to design a sequence of transformations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#Applying transformation to both categorical and numerical attributes

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(type(housing_prepared))

#Select and Train a Model

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#checking the predictions

some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:",list(some_labels))

#Calculating RMSE

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#Better Evaluation Using Cross-Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring ="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_reg_score = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                   scoring="neg_mean_squared_error", cv=10)
forest_reg_rmse_scores = np.sqrt(-forest_reg_score)
display_scores(forest_reg_rmse_scores)

# from sklearn.externals import joblib
# def save_model(model,name):
#     joblib.dump(model,name)
#     my_model_loaded = joblib.load(name)

#Fine tune your model

#Grid Search

#finding the best possible combinations of hyperparameter

from sklearn.model_selection import GridSearchCV
param_grid =[
    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[3,10], 'max_features':[2,3,4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)

#Randomized Search
"""The grid search approach is fine when you are exploring relatively few combinations, 
but when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV
CLASS. it evaluates a given number of random combinations by selctin a random value
for each hyperparameter at every iteration"""

"""
Emsemble Methods
Another way to fine-tune your system is to try to combine the models that perform best. 
the group (or "ensemble") will often perform better than the best individual model just like 
Random Forests perform better than the individual Decision Trees.
"""


#Analyze the Best Models and their Errors

feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
print(housing_prepared.shape)

extra_attrib = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attrib + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#Evaluate Your System on the Test Set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predications)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

