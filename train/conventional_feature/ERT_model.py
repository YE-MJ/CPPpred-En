import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"


mcc_values_ert = []
accuracy_values_ert = []
specificity_values_ert = []
sensitivity_values_ert = []
best_params_list_ert = []
best_mcc_list_ert = []
best_accuracy_list_ert = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name) 
    
    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_ert = ExtraTreesClassifier()

    param_distributions_ert = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_ert = RandomizedSearchCV(model_ert, param_distributions_ert, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_ert.fit(X, y)

    best_model_ert = random_search_ert.best_estimator_
    y_pred_ert = cross_val_predict(best_model_ert, X, y, cv=5)
    
    mcc_ert = matthews_corrcoef(y, y_pred_ert)
    mcc_values_ert.append(mcc_ert)

    accuracy_ert = accuracy_score(y, y_pred_ert)
    accuracy_values_ert.append(accuracy_ert)

    tn, fp, fn, tp = confusion_matrix(y, y_pred_ert).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_ert.append(specificity)
    sensitivity_values_ert.append(sensitivity)

    best_params_list_ert.append(random_search_ert.best_params_)
    best_mcc_list_ert.append(mcc_ert)
    best_accuracy_list_ert.append(accuracy_ert)

    print(f"File '{file_path}' MCC (ERT): {mcc_ert}, Accuracy (ERT): {accuracy_ert}, Best Params (ERT): {random_search_ert.best_params_}")

output_file_path = "./extratrees_feature_CPP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_ert, mcc_ert, accuracy_ert, spec_ert, sens_ert in zip(dataset_names, best_params_list_ert, best_mcc_list_ert, accuracy_values_ert, specificity_values_ert, sensitivity_values_ert):
        f.write(f"Dataset: {dataset}\nParams: {params_ert}\nMCC: {mcc_ert}\nAccuracy: {accuracy_ert}\nSpecificity: {spec_ert}\nSensitivity: {sens_ert}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters for ERT saved to '{output_file_path}'")
