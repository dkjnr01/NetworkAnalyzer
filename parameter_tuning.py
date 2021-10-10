from modules.customKits import prepareDataset

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
import pandas as pd

# read dataset
df = prepareDataset("MaliciousDoH.csv")

# set X and Y variables
X = df[0]
y = df[1]

# model paramaters to tune
model_parameters = {
    'DT': {
        'model': DecisionTreeClassifier(),
        'parameters': {
            'max_depth': [5, 10, 15, 20, 25, 30],
            'criterion': ['entropy', 'gini']
        }
    },
    'RF': {
        'model': RandomForestClassifier(),
        'parameters': {
            'n_estimators': [100, 150, 200, 250, 300],
            'max_depth': [5, 10, 15, 20],
            'criterion': ['entropy', 'gini']
        }
    },

    'MLP': {
        'model': MLPClassifier(),
        'parameters': {
            'hidden_layer_sizes': [25, 50, 75, 100],
            'activation': ['relu', 'identity', 'tanh', 'logistic'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
    },
    'GBT': {
        'model': GradientBoostingClassifier(),
        'parameters': {
            'n_estimators': [100, 150, 200, 250, 300],
            'max_depth': [5, 10, 15, 20]
        }
    }
}
# Tune models and store results in a list
score = []

for model_name, model_config in model_parameters.items():
    model = GridSearchCV(model_config['model'], model_config['parameters'],
                         cv=5, return_train_score=False)
    model.fit(X, y)
    score.append({
        'model': model_name,
        'best parameters': model.best_params_,
        'score': model.best_score_,
    })

    print(f"{model_name} generated")

# display list in a dataframe
df = pd.DataFrame(score, columns=['model', 'best parameters', 'score'])
df.to_csv("Reports/Parameters.csv")
