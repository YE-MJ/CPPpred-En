import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix, roc_auc_score
from scipy.stats import randint
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
input_directory_path = "./All_feature/CPP_embedding_train/"
test_path = "./All_feature/CPP_embedding_test/"
mcc_values_ada = []
accuracy_values_ada = []
specificity_values_ada = []
sensitivity_values_ada = []
aucroc_values_ada = []
best_params_list_ada = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name) 
    
    test_file_path = os.path.join(test_path, file_name)
    test_df = pd.read_csv(test_file_path)
    
    df = pd.read_csv(file_path)
    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    X_test = test_df.drop(['name', 'target'], axis=1)
    y_test = test_df['target']

    model_ada = AdaBoostClassifier()

    param_distributions_ada = {
        'n_estimators': randint(50, 1000),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_ada = RandomizedSearchCV(model_ada, param_distributions_ada, n_iter=100, scoring=mcc_scorer, cv=5, random_state=42, n_jobs=-1)
    
    random_search_ada.fit(X, y)
    
    best_model_ada = random_search_ada.best_estimator_
    y_test_pred_ada = best_model_ada.predict(X_test)
    
    y_test_proba_ada = best_model_ada.predict_proba(X_test)[:, 1]

    mcc_ada = matthews_corrcoef(y_test, y_test_pred_ada)
    accuracy_ada = accuracy_score(y_test, y_test_pred_ada)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_ada).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    aucroc_ada = roc_auc_score(y_test, y_test_proba_ada)
    
    mcc_values_ada.append(mcc_ada)
    accuracy_values_ada.append(accuracy_ada)
    specificity_values_ada.append(specificity)
    sensitivity_values_ada.append(sensitivity)
    aucroc_values_ada.append(aucroc_ada)
    best_params_list_ada.append(str(random_search_ada.best_params_))  

    print(f"Test File '{test_file_path}' evaluated. MCC: {mcc_ada}, Accuracy: {accuracy_ada}, AUC-ROC: {aucroc_ada}, Best Params: {random_search_ada.best_params_}")

output_df = pd.DataFrame({
    'Dataset': dataset_names,
    'Best Params': best_params_list_ada,
    'MCC': mcc_values_ada,
    'Accuracy': accuracy_values_ada,
    'Specificity': specificity_values_ada,
    'Sensitivity': sensitivity_values_ada,
    'AUC-ROC': aucroc_values_ada
})

output_file_path = "./adaboost_feature_CPP_embedding.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'")