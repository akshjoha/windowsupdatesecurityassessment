import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_and_save_model_with_tuning(csv_path, model_path): 
    df = pd.read_csv(csv_path)
    df['release_date'] = pd.to_datetime(df['Release date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    features = ['release_year', 'release_month'] 
    X = df[features]
    y = df['Base Score']

    X = X[~y.isna()]
    y = y.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Best Model Parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    csv_path = 'combined_csv.csv'  
    model_path = 'vulnerability_model.pkl'  
    train_and_save_model_with_tuning(csv_path, model_path)
