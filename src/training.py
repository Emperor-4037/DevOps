from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, silhouette_score
import pandas as pd
from .preprocessing import get_preprocessing_pipeline

def train_model(df, target_col, problem_type, model_names, prepro_config, use_cv=True, tune_hyperparams=False):
    """
    Trains models based on configuration.
    """
    results = {}
    
    # Data Split
    if problem_type in ["Classification", "Regression"] and target_col:
        X = df.drop(columns=[target_col] + (prepro_config.get('drop', []) if prepro_config else []))
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # Clustering
        X = df.drop(columns=(prepro_config.get('drop', []) if prepro_config else []))
        X_train = X
        X_test = None 
        y_train = None
        y_test = None

    preprocessor = get_preprocessing_pipeline(prepro_config) if prepro_config else None
    
    for model_name in model_names:
        model = None
        params = {}
        
        if problem_type == "Classification":
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                params = {'model__C': [0.1, 1.0, 10.0]}
            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier()
                params = {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10, 20]}
                
        elif problem_type == "Regression":
            if model_name == "Linear Regression":
                model = LinearRegression()
                params = {} # Linear reg has few tunable params for simple grid search
            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor()
                params = {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
                
        elif problem_type == "Clustering":
            if model_name == "KMeans":
                model = KMeans(n_clusters=3) # Default
                params = {'model__n_clusters': [2, 3, 4, 5]}
            elif model_name == "Agglomerative Clustering":
                model = AgglomerativeClustering()
                params = {'model__n_clusters': [2, 3, 4]}

        if model:
            steps = []
            if preprocessor:
                steps.append(('preprocessor', preprocessor))
            steps.append(('model', model))
            
            pipeline = Pipeline(steps)
            
            trained_model = pipeline
            best_params = None
            
            if tune_hyperparams and params:
                grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1)
                grid.fit(X_train, y_train if y_train is not None else X_train)
                trained_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                trained_model.fit(X_train, y_train if y_train is not None else X_train)
            
            # Evaluation
            metrics = {}
            if problem_type == "Classification":
                preds = trained_model.predict(X_test)
                metrics['accuracy'] = accuracy_score(y_test, preds)
                metrics['report'] = classification_report(y_test, preds, output_dict=True)
            elif problem_type == "Regression":
                preds = trained_model.predict(X_test)
                metrics['mse'] = mean_squared_error(y_test, preds)
                metrics['r2'] = r2_score(y_test, preds)
            elif problem_type == "Clustering":
                labels = trained_model.named_steps['model'].labels_
                # Silhouette requires at least 2 clusters and < N samples
                try:
                    metrics['silhouette'] = silhouette_score(X_train, labels)
                except:
                    metrics['silhouette'] = "N/A"
            
            results[model_name] = {
                'metrics': metrics,
                'model': trained_model,
                'best_params': best_params,
                # Store data snippets for plotting later if needed
                'X_test_head': X_test.head().to_dict() if X_test is not None else None,
                'y_test_head': y_test.head().to_dict() if y_test is not None else None 
            }
            
    return results
