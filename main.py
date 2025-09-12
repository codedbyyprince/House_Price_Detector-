import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# 1. reading the data
housing = pd.read_csv('housing.csv')

#2. creating a startified test set 
housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins = [0.0, 1.5, 3.0, 4.5, 6.0,  np.inf],
                               labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index , test_index in split.split(housing, housing['income_cat']):
    strat_train = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test = housing.loc[test_index].drop("income_cat", axis=1)


housing = strat_train.copy()

#3. seprating labels and features
housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis=1)

# print(housing, housing_labels)

# 4 seprte numerical and categorial columns 
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_atribs = ["ocean_proximity"]

# lets make the pipline 
num_pipline = Pipeline([
    ("imputer" , SimpleImputer(strategy="median")),
    ("scaler" , StandardScaler())
])

cat_pipline = Pipeline([
    ("onehot" , OneHotEncoder(handle_unknown="ignore"))
])

# constructing full pipline 
full_pipline = ColumnTransformer([
    ("num" , num_pipline, num_attribs),
    ("cat", cat_pipline, cat_atribs)    
    ])

# 6. transform the data
housing_prepared = full_pipline.fit_transform(housing)
# print(housing_prepared)


# 7. train the model 

# linear regresion 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared , housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels , lin_preds)
print(f"linearregresion = {lin_rmse}")

# decison tree  regresion 
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared , housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_labels , dec_preds)
print(f"decisiontree = {dec_rmse}")

# random forest regresion 
ran_reg = RandomForestRegressor()
ran_reg.fit(housing_prepared , housing_labels)
ran_preds = ran_reg.predict(housing_prepared)
ran_rmse = root_mean_squared_error(housing_labels , ran_preds)
print(f"randomforest = {ran_rmse}")

