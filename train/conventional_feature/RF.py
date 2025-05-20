import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"
test_path = "./All_feature/data_CPP_test/"

mcc_values_rf = []
accuracy_values_rf = []
specificity_values_rf = []
sensitivity_values_rf = []
aucroc_values_rf = []
best_params_list_rf = []
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
    
    model_rf = RandomForestClassifier()

    param_distributions_rf = {
        'n_estimators': randint(100, 1000),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_rf = RandomizedSearchCV(model_rf, param_distributions_rf, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)
    
    random_search_rf.fit(X, y)
    
    best_model_rf = random_search_rf.best_estimator_
    y_test_pred_rf = best_model_rf.predict(X_test)
    
    y_test_proba_rf = best_model_rf.predict_proba(X_test)[:, 1]

    mcc_rf = matthews_corrcoef(y_test, y_test_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_rf).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_rf = roc_auc_score(y_test, y_test_proba_rf)
    
    mcc_values_rf.append(mcc_rf)
    accuracy_values_rf.append(accuracy_rf)
    specificity_values_rf.append(specificity)
    sensitivity_values_rf.append(sensitivity)
    aucroc_values_rf.append(aucroc_rf)
    best_params_list_rf.append(str(random_search_rf.best_params_)) 

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_rf}, Accuracy: {accuracy_rf}, AUC-ROC: {aucroc_rf}, Best Params: {random_search_rf.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_rf,
    'MCC': mcc_values_rf,
    'Accuracy': accuracy_values_rf,
    'Specificity': specificity_values_rf,
    'Sensitivity': sensitivity_values_rf,
    'AUC-ROC': aucroc_values_rf
})

output_file_path = "./randomforest_feature_CPP.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
