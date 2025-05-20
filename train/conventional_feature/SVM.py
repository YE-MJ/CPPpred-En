import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import uniform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"
test_path = "./All_feature/data_CPP_test/"

mcc_values_svc = []
accuracy_values_svc = []
specificity_values_svc = []
sensitivity_values_svc = []
aucroc_values_svc = []
best_params_list_svc = []
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
    
    model_svc = SVC(probability=True)

    param_distributions_svc = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1),
        'kernel': ['linear', 'rbf']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_svc = RandomizedSearchCV(model_svc, param_distributions_svc, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)
    
    random_search_svc.fit(X, y)
    
    best_model_svc = random_search_svc.best_estimator_
    y_test_pred_svc = best_model_svc.predict(X_test)
    
    y_test_proba_svc = best_model_svc.predict_proba(X_test)[:, 1]

    mcc_svc = matthews_corrcoef(y_test, y_test_pred_svc)
    accuracy_svc = accuracy_score(y_test, y_test_pred_svc)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_svc).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_svc = roc_auc_score(y_test, y_test_proba_svc)
    
    mcc_values_svc.append(mcc_svc)
    accuracy_values_svc.append(accuracy_svc)
    specificity_values_svc.append(specificity)
    sensitivity_values_svc.append(sensitivity)
    aucroc_values_svc.append(aucroc_svc)
    best_params_list_svc.append(str(random_search_svc.best_params_)) 

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_svc}, Accuracy: {accuracy_svc}, AUC-ROC: {aucroc_svc}, Best Params: {random_search_svc.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_svc,
    'MCC': mcc_values_svc,
    'Accuracy': accuracy_values_svc,
    'Specificity': specificity_values_svc,
    'Sensitivity': sensitivity_values_svc,
    'AUC-ROC': aucroc_values_svc
})

output_file_path = "./svc_feature_CPP.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
