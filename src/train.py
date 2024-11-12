"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


max_iter = 1000
test_size = 0.3


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    return col_transf, X_train, X_test, y_train, y_test

def train(X_train, y_train, model_type):
    """
    Train a model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        TrainedModel:  model
    """

    if model_type == "logistic":
        model = LogisticRegression(max_iter=max_iter)
    elif model_type == "knn":
        k_range = list(range(1, 31))
        param_grid = dict(n_neighbors=k_range)

        # defining parameter range
        knn = KNeighborsClassifier()
        grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
        
        # fitting the model for grid search
        grid_search=grid.fit(X_train, y_train)
        num_best = grid_search.best_params_['n_neighbors']
        mlflow.log_param("best_n_neighbors", num_best)

        model = KNeighborsClassifier(n_neighbors= num_best )

    model.fit(X_train, y_train)

    # Infer signature (input and output schema)
    signature = mlflow.models.signature.infer_signature(
        X_train, model.predict(X_train)
    )

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name="churn-model",
        input_example=X_train.iloc[:3],
    )

    return model


def main():
    experiment_name = "churn-exp"
    run_name = "churn-knn"
    data_file = "data/Churn_Modelling.csv"

    model_type = "knn"  # ['knn', 'logistic']

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", test_size)

        df = pd.read_csv(data_file)
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        model = train(X_train, y_train, model_type=model_type)
        y_pred = model.predict(X_test)
        print(f"Accuracy score: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision score: {precision_score(y_test, y_pred):.2f}")
        print(f"Recall score: {recall_score(y_test, y_pred):.2f}")
        print(f"F1 score: {f1_score(y_test, y_pred):.2f}")

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()

        # Confusion matrix values
        tp = conf_mat[0][0]
        tn = conf_mat[1][1]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]

        mlflow.log_metric("true_positive", tp)
        mlflow.log_metric("true_negative", tn)
        mlflow.log_metric("false_positive", fp)
        mlflow.log_metric("false_negative", fn)

        plt.savefig("metrics/confusion-matrix.png")
        mlflow.log_artifact("metrics/confusion-matrix.png")

        plt.show()

if __name__ == "__main__":
    main()