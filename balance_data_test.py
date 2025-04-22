import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

test_path = "./CPP_feature/CPP_test/"
save_path = "./selected_weight/CPP/"  # 저장된 모델 가중치 파일 경로

# 모델 로드
models = {
    "model_cat1": joblib.load(os.path.join(save_path, "model_cat1_TPC type 1_best_model.pkl")),
    "model_ERT2": joblib.load(os.path.join(save_path, "model_ERT2_prot_t5_xl_bfd_best_model.pkl")),
    "model_ERT3": joblib.load(os.path.join(save_path, "model_ERT3_esm2_best_model.pkl")),
    "model_ERT4": joblib.load(os.path.join(save_path, "model_ERT4_esm1b_best_model.pkl")),
    "model_gb1": joblib.load(os.path.join(save_path, "model_gb1_CTDC_best_model.pkl")),
    "model_gb2": joblib.load(os.path.join(save_path, "model_gb2_esm1v_best_model.pkl"))
}

# 테스트 파일 목록과 모델 매칭
test_files = {
    "model_cat1": "TPC type 1.csv",
    "model_ERT2": "prot_t5_xl_bfd.csv",
    "model_ERT3": "esm2.csv",
    "model_ERT4": "esm1b.csv",
    "model_gb1": "CTDC.csv",
    "model_gb2": "esm1v.csv"
}

# 테스트 데이터 로드 및 레이블 설정
test_data = {}
labels = None

for model_name, test_file in test_files.items():
    file_path = os.path.join(test_path, test_file)
    data = pd.read_csv(file_path)
    
    if labels is None:
        labels = data['target'].values
    
    test_data[model_name] = data.drop(['name', 'target'], axis=1)

# 모델 확률 예측 저장
model_probabilities = []

for model_name in models:
    model = models[model_name]
    X_test = test_data[model_name]
    
    # 확률 예측 수행
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        model_probabilities.append(probs)
    else:
        raise ValueError(f"Model {model_name} does not support predict_proba.")

# 소프트 보팅 기반 앙상블 확률 계산
ensemble_proba = np.mean(model_probabilities, axis=0)
final_predictions = (ensemble_proba >= 0.5).astype(int)

# 앙상블 모델 성능 평가
final_accuracy = accuracy_score(labels, final_predictions)
final_mcc = matthews_corrcoef(labels, final_predictions)
tn, fp, fn, tp = confusion_matrix(labels, final_predictions).ravel()
final_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
final_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# 앙상블 모델 성능 출력
print("\nEnsemble Model Performance Metrics:")
print(f"  Accuracy: {final_accuracy:.4f}")
print(f"  Sensitivity: {final_sensitivity:.4f}")
print(f"  Specificity: {final_specificity:.4f}")
print(f"  MCC: {final_mcc:.4f}")
