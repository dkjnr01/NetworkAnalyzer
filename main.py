from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from time import time
import pandas as pd

# custom module that retrieves the cleaned dataset
from modules.DataPrep import prepareDataset

dataset = prepareDataset("MaliciousDoH.csv")
X = dataset[0]
y = dataset[1]

# defines the stacking ensemble model


def stack_classifier():
    # defines the base model list
    base_models = []
    base_models.append(('rf', RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)))
    base_models.append(('dt', DecisionTreeClassifier(
        max_depth=15, random_state=42, criterion="entropy")))

    # defines the meta-classifier
    meta_learner = MLPClassifier(hidden_layer_sizes=(
        100), activation="relu", max_iter=500, learning_rate='invscaling')

    # defines the stack classifier
    model = StackingClassifier(
        estimators=base_models, final_estimator=meta_learner, cv=10)

    return model

# defines all machine learning algorithms implemented


def machine_learning_models():
    models = dict()
    models['SVC'] = SVC(max_iter=-1, random_state=42)

    models['GBT'] = GradientBoostingClassifier(
        n_estimators=150, random_state=42, max_depth=15)

    models['LR'] = LogisticRegression(
        penalty='l2', max_iter=1000, random_state=42, solver='lbfgs')

    models['MLP'] = MLPClassifier(hidden_layer_sizes=(
        100), activation="relu", max_iter=1000, random_state=42, learning_rate='invscaling')

    models['Stacked'] = stack_classifier()

    return models

# standardize the scale of the dataset


def standardize(X):
    x_scaled = StandardScaler()
    X = x_scaled.fit_transform(X)
    return X

# splits the data into train and test set


def split_data_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y)

    return (X_train, X_test, y_train, y_test)

# implement feature selection using mutual information gain


def shannon_entropy(X_train, X_test, y_train):

    select_feature = SelectKBest(
        mutual_info_classif, k=19).fit(X_train, y_train)

    X_train = select_feature.transform(X_train)
    X_test = select_feature.transform(X_test)

    return (X_train, X_test)


# retrieve machine learning models
models = machine_learning_models()

# check for models that need feature scaling
model_check = ['SVC', 'MLP', 'LR']

for name, model in models.items():
    X = X
    if name in model_check:
        # if conditions are met, scale the data

        X = standardize(X)

    # split data into train and test set
    train_test = split_data_train_test(X, y)
    X_train = train_test[0]
    X_test = train_test[1]
    y_train = train_test[2]
    y_test = train_test[3]

    # implement feature selection
    featureSelection = shannon_entropy(X_train, X_test, y_train)
    X_train = featureSelection[0]
    X_test = featureSelection[1]

    # start timer
    start_timer = time()
    model = model.fit(X_train, y_train)

    # define prediction
    y_pred = model.predict(X_test)

    # assign reports to variables
    accuracy = model.score(X_test, y_test) * 100
    confusionMatrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cv_score = cross_val_score(model, X, y, cv=5, scoring="recall_macro")

    stop_timer = time() - start_timer

    # generates report of the models on a file named report.txt
    filename = f'Reports/{name}_report.txt'
    with open(filename, 'w+', encoding="utf-8") as f:
        f.write(
            f"""Accuracy: {accuracy}% \n\nConfustion Matrix \n{confusionMatrix}
        \n\nClassificiation Report \n{report} \n\nCompletion Time : {int(stop_timer)}s\n\nCross val score :{cv_score}\n{cv_score.mean()}\n\nStandard Deviation : {cv_score.std()}""")

    print(f"{name}_report generated")

print("Execution complete. All reports generated")
