# ML Models Directory

This directory is used to store trained machine learning models for the Alpaca Trading Bot.

## Model Types

The following types of models may be stored here:

- Supervised learning models (Random Forest, SVM, Neural Networks)
- Reinforcement learning models
- Deep learning models
- NLP models
- Ensemble models

## File Naming Convention

Models are saved with the following naming convention:

```
{model_type}_model.pkl
```

For example:
- `ensemble_model.pkl`
- `supervised_model.pkl`
- `reinforcement_model.pkl`
- `nlp_model.pkl`

## Usage

Models in this directory are automatically loaded by the MLStrategy class when the trading bot starts.

## Training New Models

To train and save new models, you can use the functions in the `src/ml_models.py` module:

```python
from src.ml_models import get_ml_model, prepare_training_data

# Prepare training data
X_train, X_test, y_train, y_test = prepare_training_data(historical_data)

# Get a model
model = get_ml_model('supervised')

# Train the model
model.train(X_train, y_train)

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Save the model
model.save('models/supervised_model.pkl')
```

## Future Enhancements

Future enhancements to the ML capabilities may include:
- Automated model retraining
- Model versioning
- A/B testing of different models
- Hyperparameter optimization
- Feature importance analysis
- Model explainability tools