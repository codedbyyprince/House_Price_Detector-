import joblib
import os 
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
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPLINE_FILE = "pipline.pkl"

def build_pipline(num_attribs, cat_atribs):
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
    return full_pipline

if not os.path.exists(MODEL_FILE):
    # training the model 
    housing = pd.read_csv('housing.csv')

    #2. creating a startified test set 
    housing['income_cat'] = pd.cut(housing['median_income'], 
                                bins = [0.0, 1.5, 3.0, 4.5, 6.0,  np.inf],
                                labels = [1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index , test_index in split.split(housing, housing['income_cat']):
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_atribs = ["ocean_proximity"]

    pipline = build_pipline(num_attribs, cat_atribs)
    housing_prepared = pipline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model , MODEL_FILE)
    joblib.dump(pipline , PIPLINE_FILE)

    print("CONGRATS! MODEL IS TRAINED ")