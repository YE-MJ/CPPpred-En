import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
input_directory_path = "./All_feature/CPP_embedding_train/"

mcc_values_gb = []
accuracy_values_gb = []
specificity_values_gb = []
sensitivity_values_gb = []
best_params_list_gb = []
best_mcc_list_gb = []
best_accuracy_list_gb = []
dataset_names = []

csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name) 
    
    df = pd.read_csv(file_path)

    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_gb = GradientBoostingClassifier()

    param_distributions_gb = {
        'n_estimators': randint(50, 500),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': randint(3, 10),
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_gb = RandomizedSearchCV(model_gb, param_distributions_gb, n_iter=100, scoring=mcc_scorer, cv=5, verbose=1, n_jobs=-1, random_state=42)
    random_search_gb.fit(X, y)

    best_model_gb = random_search_gb.best_estimator_
    y_pred_gb = cross_val_predict(best_model_gb, X, y, cv=5)
    
    mcc_gb = matthews_corrcoef(y, y_pred_gb)
    mcc_values_gb.append(mcc_gb)

    accuracy_gb = accuracy_score(y, y_pred_gb)
    accuracy_values_gb.append(accuracy_gb)

    tn, fp, fn, tp = confusion_matrix(y, y_pred_gb).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_gb.append(specificity)
    sensitivity_values_gb.append(sensitivity)

    best_params_list_gb.append(random_search_gb.best_params_)
    best_mcc_list_gb.append(mcc_gb)
    best_accuracy_list_gb.append(accuracy_gb)

    print(f"File '{file_path}' MCC (GB): {mcc_gb}, Accuracy (GB): {accuracy_gb}, Specificity (GB): {specificity}, Sensitivity (GB): {sensitivity}")

output_file_path = "./gradientboosting_feature_CPP_embedding.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_gb, mcc_gb, accuracy_gb, spec_gb, sens_gb in zip(dataset_names, best_params_list_gb, best_mcc_list_gb, accuracy_values_gb, specificity_values_gb, sensitivity_values_gb):
        f.write(f"Dataset: {dataset}\nParams: {params_gb}\nMCC: {mcc_gb}\nAccuracy: {accuracy_gb}\nSpecificity: {spec_gb}\nSensitivity: {sens_gb}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters for GB saved to '{output_file_path}'")
