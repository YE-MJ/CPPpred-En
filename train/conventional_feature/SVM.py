import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
input_directory_path = "./All_feature/data_CPP_train/"

mcc_values_svc = []
accuracy_values_svc = []
specificity_values_svc = []
sensitivity_values_svc = []
best_params_list_svc = []
best_mcc_list_svc = []
best_accuracy_list_svc = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)

    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_svc = SVC()

    param_distributions_svc = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1),
        'kernel': ['linear', 'rbf']
    }
    
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_svc = RandomizedSearchCV(model_svc, param_distributions_svc, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_svc.fit(X, y)

    best_model_svc = random_search_svc.best_estimator_
    y_pred_svc = cross_val_predict(best_model_svc, X, y, cv=5)

    cm = confusion_matrix(y, y_pred_svc)
    tn, fp, fn, tp = cm.ravel()

    mcc_svc = matthews_corrcoef(y, y_pred_svc)
    mcc_values_svc.append(mcc_svc)

    accuracy_svc = accuracy_score(y, y_pred_svc)
    accuracy_values_svc.append(accuracy_svc)

    specificity = tn / (tn + fp)
    specificity_values_svc.append(specificity)

    sensitivity = tp / (tp + fn)
    sensitivity_values_svc.append(sensitivity)

    best_params_list_svc.append(random_search_svc.best_params_)

    print(f"파일 '{file_path}' MCC (SVC): {mcc_svc}, Accuracy (SVC): {accuracy_svc}, Specificity (Specificity): {specificity}, Sensitivity (Sensitivity): {sensitivity}")

output_file_path = "./svc_feature_CPP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_svc, mcc_svc, accuracy_svc, specificity_svc, sensitivity_svc in zip(dataset_names, best_params_list_svc, mcc_values_svc, accuracy_values_svc, specificity_values_svc, sensitivity_values_svc):
        f.write(f"Dataset: {dataset}\nParams: {params_svc}\nMCC: {mcc_svc}\nAccuracy: {accuracy_svc}\nSpecificity: {specificity_svc}\nSensitivity: {sensitivity_svc}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
