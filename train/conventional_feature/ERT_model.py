import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"
test_path = "./All_feature/data_CPP_test/"

mcc_values_ert = []
accuracy_values_ert = []
specificity_values_ert = []
sensitivity_values_ert = []
aucroc_values_ert = []
best_params_list_ert = []
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
    
    model_ert = ExtraTreesClassifier()

    param_distributions_ert = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_ert = RandomizedSearchCV(model_ert, param_distributions_ert, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)
    
    random_search_ert.fit(X, y)
    
    best_model_ert = random_search_ert.best_estimator_
    y_test_pred_ert = best_model_ert.predict(X_test)
    
    y_test_proba_ert = best_model_ert.predict_proba(X_test)[:, 1]

    mcc_ert = matthews_corrcoef(y_test, y_test_pred_ert)
    accuracy_ert = accuracy_score(y_test, y_test_pred_ert)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_ert).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_ert = roc_auc_score(y_test, y_test_proba_ert)
    
    mcc_values_ert.append(mcc_ert)
    accuracy_values_ert.append(accuracy_ert)
    specificity_values_ert.append(specificity)
    sensitivity_values_ert.append(sensitivity)
    aucroc_values_ert.append(aucroc_ert)
    best_params_list_ert.append(str(random_search_ert.best_params_)) 

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_ert}, Accuracy: {accuracy_ert}, AUC-ROC: {aucroc_ert}, Best Params: {random_search_ert.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_ert,
    'MCC': mcc_values_ert,
    'Accuracy': accuracy_values_ert,
    'Specificity': specificity_values_ert,
    'Sensitivity': sensitivity_values_ert,
    'AUC-ROC': aucroc_values_ert
})

output_file_path = "./extratrees_feature_CPP.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
