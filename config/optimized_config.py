# Optimized configuration for ML models and ensemble weights

# RandomForest parameters
RF_PARAMS = {'class_weight': None, 'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}

# GradientBoosting parameters
GB_PARAMS = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}

# Ensemble weights
ENSEMBLE_WEIGHTS = {'gradient_boosting': 0.0, 'random_forest': 1.0}
