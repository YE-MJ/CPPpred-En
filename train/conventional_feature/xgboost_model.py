import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"
test_path = "./All_feature/data_CPP_test/"

mcc_values_xgb = []
accuracy_values_xgb = []
specificity_values_xgb = []
sensitivity_values_xgb = []
aucroc_values_xgb = []
best_params_list_xgb = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)

    test_file_path = os.path.join(test_path, file_name)
    
    if not os.path.exists(test_file_path):
        print(f"Test file '{test_file_path}' does not exist. Skipping this dataset.")
        continue
    
    df = pd.read_csv(file_path)
    test_df = pd.read_csv(test_file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']
    X_test = test_df.drop(['name', 'target'], axis=1)
    y_test = test_df['target']
    
    model_xgb = XGBClassifier(eval_metric='logloss')

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
    
    random_search_xgb = RandomizedSearchCV(model_xgb, param_distributions_xgb, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)
    
    random_search_xgb.fit(X, y)
    
    best_model_xgb = random_search_xgb.best_estimator_
    y_test_pred_xgb = best_model_xgb.predict(X_test)
    
    y_test_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1]

    mcc_xgb = matthews_corrcoef(y_test, y_test_pred_xgb)
    accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_xgb).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_xgb = roc_auc_score(y_test, y_test_proba_xgb)
    
    mcc_values_xgb.append(mcc_xgb)
    accuracy_values_xgb.append(accuracy_xgb)
    specificity_values_xgb.append(specificity)
    sensitivity_values_xgb.append(sensitivity)
    aucroc_values_xgb.append(aucroc_xgb)
    best_params_list_xgb.append(str(random_search_xgb.best_params_)) 

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_xgb}, Accuracy: {accuracy_xgb}, AUC-ROC: {aucroc_xgb}, Best Params: {random_search_xgb.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_xgb,
    'MCC': mcc_values_xgb,
    'Accuracy': accuracy_values_xgb,
    'Specificity': specificity_values_xgb,
    'Sensitivity': sensitivity_values_xgb,
    'AUC-ROC': aucroc_values_xgb
})

output_file_path = "./xgboost_feature_CPP.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
