import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
input_directory_path = "./All_feature/CPP_embedding_train/"

mcc_values_rf = []
accuracy_values_rf = []
specificity_values_rf = []
sensitivity_values_rf = []
best_params_list_rf = []
best_mcc_list_rf = []
best_accuracy_list_rf = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)  

    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']


    model_rf = RandomForestClassifier()

    param_distributions_rf = {
        'n_estimators': randint(100, 1000),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_rf = RandomizedSearchCV(model_rf, param_distributions_rf, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_rf.fit(X, y)

    best_model_rf = random_search_rf.best_estimator_
    y_pred_rf = cross_val_predict(best_model_rf, X, y, cv=5)

    cm = confusion_matrix(y, y_pred_rf)
    tn, fp, fn, tp = cm.ravel()

    mcc_rf = matthews_corrcoef(y, y_pred_rf)
    mcc_values_rf.append(mcc_rf)

    accuracy_rf = accuracy_score(y, y_pred_rf)
    accuracy_values_rf.append(accuracy_rf)

    specificity = tn / (tn + fp)
    specificity_values_rf.append(specificity)

    sensitivity = tp / (tp + fn)
    sensitivity_values_rf.append(sensitivity)

    best_params_list_rf.append(random_search_rf.best_params_)
    best_mcc_list_rf.append(mcc_rf)
    best_accuracy_list_rf.append(accuracy_rf)

    print(f"파일 '{file_path}' MCC (RandomForest): {mcc_rf}, Accuracy (RandomForest): {accuracy_rf}, Specificity (Specificity): {specificity}, Sensitivity (Sensitivity): {sensitivity}")

output_file_path = "./randomforest_feature_CPP_embedding.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_rf, mcc_rf, accuracy_rf, specificity_rf, sensitivity_rf in zip(dataset_names, best_params_list_rf, best_mcc_list_rf, accuracy_values_rf, specificity_values_rf, sensitivity_values_rf):
        f.write(f"Dataset: {dataset}\nParams: {params_rf}\nMCC: {mcc_rf}\nAccuracy: {accuracy_rf}\nSpecificity: {specificity_rf}\nSensitivity: {sensitivity_rf}\n\n")

print(f"MCC 및 Accuracy 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
