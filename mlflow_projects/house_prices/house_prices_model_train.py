"""House Pricing Model - Track and Deploy with MLFlow"""

# Imports
import logging
import mlflow

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from numpy import savetxt

import pandas as pd
import numpy as np
import xgboost as xgb



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

## Register to the tracking server and name the expirement
EXPERIMENT_NAME = "MLProject_house_prices_sklearn_pipe_xgb"
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)


## Function to evaluate results of regression model
def eval_metrics(actual, pred):
    """Function to evaluate results"""
    
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


## Simplify the MLFlow example by reducing the number of columns
def minimize_dataset(df):
    """ 
    Return a minimized version of the data, 
    to make easy to expirement with feature transformation 
    """

    new_df = pd.DataFrame()

    new_df['price'] = df.price

    # Add new number of hotel rooms to the new df
    new_df['n_hot_rooms'] = df.n_hot_rooms
    
    # Filling missing value in n_hos_beds
    new_df['n_hos_beds'] = df.n_hos_beds

    # Transform the inversely proportional functoin into a liner function using log
    new_df['crime_rate'] = df.crime_rate

    new_df['room_num'] = df.room_num
    new_df['teachers'] = df.teachers
    new_df['waterbody'] = df.waterbody
    
    # Convert categorical variables to dummy variables
    #new_df = pd.get_dummies(new_df, drop_first=True)

    return new_df


## Function to retrieve data
def get_data():
    """Get data from source and minimize it"""

    raw_dataset = pd.read_csv("~/Projects/studynotes/data/House_Price.csv", header=0)
    mini_dataset = minimize_dataset(raw_dataset)
    return mini_dataset


def get_training_testing_data(df, y):
    """ Split data into train and test """

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


## Feature preprocessing

numeric_features = ['n_hot_rooms','n_hos_beds','room_num','teachers']
numeric_transformer = Pipeline(steps=[
     ('imputer', SimpleImputer(strategy='mean')),
     ('scaler', StandardScaler())])

categorical_features = [ 'waterbody']
categorical_transformer = Pipeline(steps=[ 
     ('imputer', SimpleImputer(strategy='constant',fill_value=0)),
     ('ordinal_encoder', preprocessing.OrdinalEncoder())])

log_features = ['crime_rate']
log_transformer = Pipeline(steps=[
     #('imputer', SimpleImputer(strategy='median')),
     ('log_tranformer', FunctionTransformer(np.log1p, validate=True))
])


preprocessor = ColumnTransformer(
     transformers=[
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features),
          ('log', log_transformer, log_features)
          ]
     )


def run():
    """Execute mlflow expirement run"""
    with mlflow.start_run() as model_tracking_run:
        ## Get Training Data
        df = get_data()
        
        if 'price' in df:
            y = df.pop('price')

        X_train, X_test, y_train, y_test = get_training_testing_data(df, y)
        
        ## Construct Model 
        params = {'max_depth':12, 'n_estimators':1000, 'learning_rate':0.1, 'subsample':0.8, 'colsample_bytree':0.8}

        ## Log model params in MLFlow
        mlflow.log_params(params)

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBRegressor(**params))
            ], verbose=True)
        
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        ## Log model in MLFlow
        mlflow.sklearn.log_model(model, "preprocess_and_xgb")

        rmse, mae, r2 = eval_metrics(y_test, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        ## Save and log predicted values
        prediction_file = f"prediction_{model_tracking_run.info.run_id}.csv"
        savetxt(prediction_file, predictions)
        mlflow.log_artifact(prediction_file)

        ## Create a plot that shows predicted vs true prices
        compare = pd.DataFrame({0:y_test.values, 1:predictions })
        compare.columns=['true_price', 'predicted_price']
        fig = compare.plot(use_index=True).get_figure()
        fig_file = f"true_vs_predicted_{model_tracking_run.info.run_id}.png"
        fig.savefig(fig_file)
        mlflow.log_artifact(fig_file)



if __name__ == "__main__":
    run()
