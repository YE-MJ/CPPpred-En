import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# preprocessing_feature_extraction 디렉토리 경로 설정
input_directory_path = "./All_feature/data_CPP_train/"

mcc_values_ada = []
accuracy_values_ada = []
specificity_values_ada = []
sensitivity_values_ada = []
best_params_list_ada = []
best_mcc_list_ada = []
best_accuracy_list_ada = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)  # 파일명 리스트에 추가
    
    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_ada = AdaBoostClassifier()

    param_distributions_ada = {
        'n_estimators': randint(50, 500),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_ada = RandomizedSearchCV(model_ada, param_distributions_ada, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42)
    
    random_search_ada.fit(X, y)
    
    best_model_ada = random_search_ada.best_estimator_
    y_pred_ada = cross_val_predict(best_model_ada, X, y, cv=5)
    
    mcc_ada = matthews_corrcoef(y, y_pred_ada)
    mcc_values_ada.append(mcc_ada)

    accuracy_ada = accuracy_score(y, y_pred_ada)
    accuracy_values_ada.append(accuracy_ada)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred_ada).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_ada.append(specificity)
    sensitivity_values_ada.append(sensitivity)

    best_params_list_ada.append(random_search_ada.best_params_)
    best_mcc_list_ada.append(mcc_ada)
    best_accuracy_list_ada.append(accuracy_ada)

    print(f"File '{file_path}' MCC (Adaboost): {mcc_ada}, Accuracy (Adaboost): {accuracy_ada}, Best Params (Adaboost): {random_search_ada.best_params_}")

output_file_path = "./adaboost_feature_CPP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params, mcc, accuracy, spec, sens in zip(dataset_names, best_params_list_ada, best_mcc_list_ada, accuracy_values_ada, specificity_values_ada, sensitivity_values_ada):
        f.write(f"Dataset: {dataset}\nParams: {params}\nMCC: {mcc}\nAccuracy: {accuracy}\nSpecificity: {spec}\nSensitivity: {sens}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters saved to '{output_file_path}'")



