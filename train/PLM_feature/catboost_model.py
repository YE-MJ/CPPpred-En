import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/CPP_embedding_train/"

mcc_values_cat = []
accuracy_values_cat = []
specificity_values_cat = []
sensitivity_values_cat = []
best_params_list_cat = []
best_mcc_list_cat = []
best_accuracy_list_cat = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)  
    
    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_cat = CatBoostClassifier()

    param_distributions_cat = {
        'iterations': randint(100, 1000),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_cat = RandomizedSearchCV(model_cat, param_distributions_cat, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_cat.fit(X, y)

    best_model_cat = random_search_cat.best_estimator_
    y_pred_cat = cross_val_predict(best_model_cat, X, y, cv=5)
    
    mcc_cat = matthews_corrcoef(y, y_pred_cat)
    mcc_values_cat.append(mcc_cat)

    accuracy_cat = accuracy_score(y, y_pred_cat)
    accuracy_values_cat.append(accuracy_cat)

    tn, fp, fn, tp = confusion_matrix(y, y_pred_cat).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_cat.append(specificity)
    sensitivity_values_cat.append(sensitivity)

    best_params_list_cat.append(random_search_cat.best_params_)
    best_mcc_list_cat.append(mcc_cat)
    best_accuracy_list_cat.append(accuracy_cat)

    print(f"File '{file_path}' MCC (CatBoost): {mcc_cat}, Accuracy (CatBoost): {accuracy_cat}, Best Params (CatBoost): {random_search_cat.best_params_}")

output_file_path = "./catboost_feature_CPP_embedding.txt"
with open(output_file_path, 'w') as f:
    for dataset, params, mcc, accuracy, spec, sens in zip(dataset_names, best_params_list_cat, best_mcc_list_cat, accuracy_values_cat, specificity_values_cat, sensitivity_values_cat):
        f.write(f"Dataset: {dataset}\nParams: {params}\nMCC: {mcc}\nAccuracy: {accuracy}\nSpecificity: {spec}\nSensitivity: {sens}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters saved to '{output_file_path}'")
