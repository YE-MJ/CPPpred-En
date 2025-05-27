import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
input_directory_path = "./All_feature/CPP_embedding_train/"

mcc_values_xgb = []
accuracy_values_xgb = []
specificity_values_xgb = []
sensitivity_values_xgb = []
best_params_list_xgb = []
best_mcc_list_xgb = []
best_accuracy_list_xgb = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)

    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    param_distributions_xgb = {
        'n_estimators': randint(100, 1000),
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.1, 0.5, 1, 2]
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_xgb = RandomizedSearchCV(model_xgb, param_distributions_xgb, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_xgb.fit(X, y)

    best_model_xgb = random_search_xgb.best_estimator_
    y_pred_xgb = cross_val_predict(best_model_xgb, X, y, cv=5)

    cm = confusion_matrix(y, y_pred_xgb)
    tn, fp, fn, tp = cm.ravel()

    mcc_xgb = matthews_corrcoef(y, y_pred_xgb)
    mcc_values_xgb.append(mcc_xgb)

    accuracy_xgb = accuracy_score(y, y_pred_xgb)
    accuracy_values_xgb.append(accuracy_xgb)

    specificity = tn / (tn + fp)
    specificity_values_xgb.append(specificity)

    sensitivity = tp / (tp + fn)
    sensitivity_values_xgb.append(sensitivity)

    best_params_list_xgb.append(random_search_xgb.best_params_)

    print(f"파일 '{file_path}' MCC (XGBoost): {mcc_xgb}, Accuracy (XGBoost): {accuracy_xgb}, Specificity (Specificity): {specificity}, Sensitivity (Sensitivity): {sensitivity}")

output_file_path = "./xgboost_feature_CPP_embedding.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_xgb, mcc_xgb, accuracy_xgb, specificity_xgb, sensitivity_xgb in zip(dataset_names, best_params_list_xgb, mcc_values_xgb, accuracy_values_xgb, specificity_values_xgb, sensitivity_values_xgb):
        f.write(f"Dataset: {dataset}\nParams: {params_xgb}\nMCC: {mcc_xgb}\nAccuracy: {accuracy_xgb}\nSpecificity: {specificity_xgb}\nSensitivity: {sensitivity_xgb}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
