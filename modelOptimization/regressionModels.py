import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import argparse

class RegressionModels:
    def __init__(self, features, target, model_type='RandomForest'):
        """
        Initialize the RegressionModels class with dataset and selected model type.

        :param features: Features dataframe.
        :param target: Target dataframe.
        :param model_type: Type of model to be used for regression. Options: 'RandomForest', 'XGBoost', 'MLP'.
        """
        self.features = features
        self.target = target
        self.model_type = model_type
        self.model = self._initialize_model()

        # Split the dataset into training and testing sets (80-20 split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target, test_size=0.2, shuffle=True)

    def _initialize_model(self):
        """
        Initialize the model based on the selected model type.

        :return: A model object based on the chosen type.
        """
        if self.model_type == 'RandomForest':
            return RandomForestRegressor(n_estimators=175, max_depth=32,
                                         min_samples_leaf=8, random_state=48)
        elif self.model_type == 'XGBoost':
            return xgb.XGBRegressor(objective='reg:squarederror',
                                    n_estimators=35, random_state=48, booster='gbtree',
                                    learning_rate=0.1, max_depth=4, colsample_bytree=0.8)
        elif self.model_type == 'MLP':
            return MLPRegressor(hidden_layer_sizes=(100,),
                                max_iter=200, early_stopping=True)
        else:
            raise ValueError("Unsupported model type. Choose from 'RandomForest', 'XGBoost', 'MLP'.")

    def fit_model(self):
        """
        Fit the model to the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def cross_validate(self, cv=5):
        """
        Perform cross-validation and return the R2 score of each fold.

        :param cv: The number of folds for cross-validation. Default is 5.
        :return: R2 scores for each fold of the cross-validation.
        """
        scores = cross_val_score(self.model, self.features, self.target, cv=KFold(cv, shuffle=True))
        return scores

    def cross_val_predict_r2(self, cv=5):
        """
        Perform cross-validation and return the R2 score of the predicted values.

        :param cv: The number of folds for cross-validation. Default is 5.
        :return: R2 score of cross-validation predictions.
        """
        pred_cv = cross_val_predict(self.model, self.features, self.target, cv=KFold(cv, shuffle=True))
        r2_cv = metrics.r2_score(self.target, pred_cv)
        return r2_cv

    def evaluate_model(self):
        """
        Evaluate the trained model using R2 score on the test set.

        :return: R2 score of the model on the test set.
        """
        y_pred = self.model.predict(self.X_test)
        return metrics.r2_score(self.y_test, y_pred)

    @staticmethod
    def parse_arguments():
        """
        Parse command-line arguments for the data file and model selection.

        :return: Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="Regression models comparison.")
        parser.add_argument('data_path', type=str, help="Path to the dataset (Excel file).")
        parser.add_argument('--model', type=str, default='RandomForest', choices=['RandomForest', 'XGBoost', 'MLP'],
                            help="Choose the regression model (default: 'RandomForest').")

        return parser.parse_args()


def main():
    # Parse command-line arguments
    args = RegressionModels.parse_arguments()

    # Load the data
    database = pd.read_excel(args.data_path)
    features = database.iloc[:, :-1]  # Features
    target = database.iloc[:, -1]     # Target

    # Initialize the class with the given file path and model type
    regression_model = RegressionModels(features=features, target=target, model_type=args.model)

    # Train the model
    regression_model.fit_model()

    # Perform cross-validation and evaluation as usual
    cross_val_scores = regression_model.cross_validate(cv=5)
    print(f"Cross-validation R2 scores: {cross_val_scores}")

    r2_cv = regression_model.cross_val_predict_r2(cv=5)
    print(f"Cross-validation R2 score of predictions: {r2_cv}")


if __name__ == "__main__":
    main()
