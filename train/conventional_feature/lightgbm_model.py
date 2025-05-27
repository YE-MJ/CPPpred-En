import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
input_directory_path = "./All_feature/data_CPP_train/"

mcc_values_lgbm = []
accuracy_values_lgbm = []
specificity_values_lgbm = []
sensitivity_values_lgbm = []
best_params_list_lgbm = []
best_mcc_list_lgbm = []
best_accuracy_list_lgbm = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name) 

    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_lgbm = LGBMClassifier()

    param_distributions_lgbm = {
        'n_estimators': randint(100, 1000),
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.1, 0.5, 1, 2]
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_lgbm = RandomizedSearchCV(model_lgbm, param_distributions_lgbm, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_lgbm.fit(X, y)
    
    best_model_lgbm = random_search_lgbm.best_estimator_
    y_pred_lgbm = cross_val_predict(best_model_lgbm, X, y, cv=5)
    
    mcc_lgbm = matthews_corrcoef(y, y_pred_lgbm)
    mcc_values_lgbm.append(mcc_lgbm)

    accuracy_lgbm = accuracy_score(y, y_pred_lgbm)
    accuracy_values_lgbm.append(accuracy_lgbm)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred_lgbm).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_lgbm.append(specificity)
    sensitivity_values_lgbm.append(sensitivity)

    best_params_list_lgbm.append(random_search_lgbm.best_params_)
    best_mcc_list_lgbm.append(mcc_lgbm)
    best_accuracy_list_lgbm.append(accuracy_lgbm)

    print(f"File '{file_path}' MCC (LightGBM): {mcc_lgbm}, Accuracy (LightGBM): {accuracy_lgbm}, Specificity (LightGBM): {specificity}, Sensitivity (LightGBM): {sensitivity}")

output_file_path = "./lgbm_feature_CPP.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for dataset, params, mcc, accuracy, spec, sens in zip(dataset_names, best_params_list_lgbm, best_mcc_list_lgbm, accuracy_values_lgbm, specificity_values_lgbm, sensitivity_values_lgbm):
        f.write(f"Dataset: {dataset}\nParams: {params}\nMCC: {mcc}\nAccuracy: {accuracy}\nSpecificity: {spec}\nSensitivity: {sens}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
