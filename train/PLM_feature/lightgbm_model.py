import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/CPP_embedding_train/"
test_path = "./All_feature/CPP_embedding_test/"

mcc_values_lgbm = []
accuracy_values_lgbm = []
specificity_values_lgbm = []
sensitivity_values_lgbm = []
aucroc_values_lgbm = []
best_params_list_lgbm = []
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
    
    model_lgbm = LGBMClassifier(verbose=-1)

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
    
    random_search_lgbm = RandomizedSearchCV(model_lgbm, param_distributions_lgbm, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42)
    
    random_search_lgbm.fit(X, y)
    
    best_model_lgbm = random_search_lgbm.best_estimator_
    y_test_pred_lgbm = best_model_lgbm.predict(X_test)
    
    y_test_proba_lgbm = best_model_lgbm.predict_proba(X_test)[:, 1]

    mcc_lgbm = matthews_corrcoef(y_test, y_test_pred_lgbm)
    accuracy_lgbm = accuracy_score(y_test, y_test_pred_lgbm)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_lgbm).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_lgbm = roc_auc_score(y_test, y_test_proba_lgbm)
    
    mcc_values_lgbm.append(mcc_lgbm)
    accuracy_values_lgbm.append(accuracy_lgbm)
    specificity_values_lgbm.append(specificity)
    sensitivity_values_lgbm.append(sensitivity)
    aucroc_values_lgbm.append(aucroc_lgbm)
    best_params_list_lgbm.append(str(random_search_lgbm.best_params_))

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_lgbm}, Accuracy: {accuracy_lgbm}, AUC-ROC: {aucroc_lgbm}, Best Params: {random_search_lgbm.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_lgbm,
    'MCC': mcc_values_lgbm,
    'Accuracy': accuracy_values_lgbm,
    'Specificity': specificity_values_lgbm,
    'Sensitivity': sensitivity_values_lgbm,
    'AUC-ROC': aucroc_values_lgbm
})

output_file_path = "./lgbm_feature_CPP_embedding.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
