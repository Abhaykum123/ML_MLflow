from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.data

# Load the Breast Cancer dataset
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name = 'target')

# Splitting into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter gird for GridSearchCV
param_gird = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_gird, cv = 5, n_jobs= -1, verbose=2)

# # Run without MLflow from here
# grid_search.fit(x_train, y_train)

# # Displaying the best params and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(best_params)
# print(best_score)

mlflow.set_experiment('breast_cancer_rf_hp')

with mlflow.start_run() as parent:
    grid_search.fit(x_train, y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric('accuracy', grid_search.cv_results_['mean_test_score'][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric('cv_accuracy', best_score)

    # Log training data
    train_df = x_train.copy()
    train_df['target'] = y_train.values

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, context='training')


    # Log test data
    test_df = x_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,'test')

     # Log best model
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        artifact_path='random_forest_best_model'
    )

    # Set metadata tags
    mlflow.set_tags({
        'author': 'Abhay Kumar',
        'project': 'Breast Cancer Classification',
        'algorithm': 'RandomForest + GridSearchCV'
    })


    # Print best results
    print(best_params)
    print(best_score)