import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from typing import Tuple, Dict, Any

class UrbanModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor()
        }
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42)
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           param_grid: Dict[str, Any] = None) -> RandomForestRegressor:
        """Train Random Forest model with optional hyperparameter tuning."""
        if param_grid:
            grid_search = GridSearchCV(
                RandomForestRegressor(),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            self.models['random_forest'] = grid_search.best_estimator_
        else:
            self.models['random_forest'].fit(X_train, y_train)
        return self.models['random_forest']
        
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               param_grid: Dict[str, Any] = None) -> GradientBoostingRegressor:
        """Train Gradient Boosting model with optional hyperparameter tuning."""
        if param_grid:
            grid_search = GridSearchCV(
                GradientBoostingRegressor(),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            self.models['gradient_boosting'] = grid_search.best_estimator_
        else:
            self.models['gradient_boosting'].fit(X_train, y_train)
        return self.models['gradient_boosting']
        
    def evaluate_model(self, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        model = self.models[model_name]
        y_pred = model.predict(X_test)
            
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
        
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Log model metrics using MLflow."""
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.models[model_name], model_name) 