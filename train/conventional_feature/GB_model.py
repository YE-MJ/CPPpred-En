import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./All_feature/data_CPP_train/"
test_path = "./All_feature/data_CPP_test/"

mcc_values_gb = []
accuracy_values_gb = []
specificity_values_gb = []
sensitivity_values_gb = []
aucroc_values_gb = []
best_params_list_gb = []
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
    
    model_gb = GradientBoostingClassifier()

    param_distributions_gb = {
        'n_estimators': randint(50, 500),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': randint(3, 10),
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_gb = RandomizedSearchCV(model_gb, param_distributions_gb, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)

    random_search_gb.fit(X, y)

    best_model_gb = random_search_gb.best_estimator_
    y_test_pred_gb = best_model_gb.predict(X_test)
    
    y_test_proba_gb = best_model_gb.predict_proba(X_test)[:, 1]

    mcc_gb = matthews_corrcoef(y_test, y_test_pred_gb)
    accuracy_gb = accuracy_score(y_test, y_test_pred_gb)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_gb).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_gb = roc_auc_score(y_test, y_test_proba_gb)
    
    mcc_values_gb.append(mcc_gb)
    accuracy_values_gb.append(accuracy_gb)
    specificity_values_gb.append(specificity)
    sensitivity_values_gb.append(sensitivity)
    aucroc_values_gb.append(aucroc_gb)
    best_params_list_gb.append(str(random_search_gb.best_params_)) 
    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_gb}, Accuracy: {accuracy_gb}, AUC-ROC: {aucroc_gb}, Best Params: {random_search_gb.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_gb,
    'MCC': mcc_values_gb,
    'Accuracy': accuracy_values_gb,
    'Specificity': specificity_values_gb,
    'Sensitivity': sensitivity_values_gb,
    'AUC-ROC': aucroc_values_gb
})

output_file_path = "./gradientboosting_feature_CPP.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")
