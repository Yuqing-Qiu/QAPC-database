import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse

class HyperparameterTuningModel:
    def __init__(self, features, amount):
        """
        Initializes the HyperparameterTuningModel with features (features) and target (amount).

        :param features: Features DataFrame.
        :param amount: Target DataFrame.
        """
        self.features = features
        self.amount = amount

    def perform_grid_search(self, X_train, y_train):
        """
        Perform GridSearchCV to find the best model parameters.

        :param X_train: Training features.
        :param y_train: Training target.
        :return: Best model found by GridSearchCV.
        """
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [50, 75, 100, 125, 150, 175, 200],
            'max_depth': [8, 16, 32, 64, 128],
            'min_samples_leaf': [2, 4, 6, 8, 16],
            'criterion': ['squared_error', 'absolute_error', 'poisson']
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model using R2 score and error metrics.

        :param model: The trained model.
        :param X_test: Test features.
        :param y_test: Test target.
        :return: None
        """
        y_pred = model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        print(f"R2 Score on Test Data: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    def train_and_evaluate(self):
        """
        Split the data, perform grid search, and evaluate the model.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.amount, test_size=0.2, shuffle=True)

        # Perform GridSearchCV to get the best model
        best_model = self.perform_grid_search(X_train, y_train)

        # Evaluate the best model on the test set
        self.evaluate_model(best_model, X_test, y_test)

    @staticmethod
    def parse_arguments():
        """
        Parse command-line arguments for the data file.

        :return: Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="Regression models comparison.")
        parser.add_argument('data_path', type=str, help="Path to the dataset (Excel file).")
        return parser.parse_args()

def main():
    # Parse command-line arguments
    args = HyperparameterTuningModel.parse_arguments()

    # Load the dataset
    database = pd.read_excel(args.data_path)
    features = database.iloc[:, :-1]  # Features DataFrame
    amount = database.iloc[:, -1]   # Target DataFrame

    # Initialize and train the regression model
    hyperparameter_model = HyperparameterTuningModel(features=features, amount=amount)
    hyperparameter_model.train_and_evaluate()

if __name__ == "__main__":
    main()
