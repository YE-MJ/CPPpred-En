import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"
test_path = "./All_feature/data_CPP_test/"

mcc_values_cat = []
accuracy_values_cat = []
specificity_values_cat = []
sensitivity_values_cat = []
aucroc_values_cat = []
best_params_list_cat = []
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
    
    model_cat = CatBoostClassifier(silent=True)

    param_distributions_cat = {
        'iterations': randint(100, 1000),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_cat = RandomizedSearchCV(model_cat, param_distributions_cat, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)
    
    random_search_cat.fit(X, y)
    
    best_model_cat = random_search_cat.best_estimator_
    y_test_pred_cat = best_model_cat.predict(X_test)
    
    y_test_proba_cat = best_model_cat.predict_proba(X_test)[:, 1]

    mcc_cat = matthews_corrcoef(y_test, y_test_pred_cat)
    accuracy_cat = accuracy_score(y_test, y_test_pred_cat)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_cat).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_cat = roc_auc_score(y_test, y_test_proba_cat)
    
    mcc_values_cat.append(mcc_cat)
    accuracy_values_cat.append(accuracy_cat)
    specificity_values_cat.append(specificity)
    sensitivity_values_cat.append(sensitivity)
    aucroc_values_cat.append(aucroc_cat)
    best_params_list_cat.append(str(random_search_cat.best_params_)) 

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_cat}, Accuracy: {accuracy_cat}, AUC-ROC: {aucroc_cat}, Best Params: {random_search_cat.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_cat,
    'MCC': mcc_values_cat,
    'Accuracy': accuracy_values_cat,
    'Specificity': specificity_values_cat,
    'Sensitivity': sensitivity_values_cat,
    'AUC-ROC': aucroc_values_cat
})

output_file_path = "./catboost_feature_CPP.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
