import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from joblib import dump, load
from google.colab import drive

# ✅ Google Drive 마운트
drive.mount('/content/drive')

# ✅ 데이터 경로 설정
BASE_PATH = "/content/drive/MyDrive/hepscope/train_val_sample_test/5sample/"
RESULTS_PATH = "/content/drive/MyDrive/hepscope/results/"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ✅ 파일 로드 함수
def load_csv(file_path):
    try:
        return pd.read_csv(file_path).drop(columns=['Unnamed: 0'], errors='ignore')
    except Exception as e:
        print(f"❌ 파일 로드 실패 ({file_path}): {e}")
        return None

# ✅ 데이터 정규화 및 결합 함수
def prepare_data(normal_df, malig_df, scaler=None):
    if normal_df is None or malig_df is None:
        return None, None, None
    normal_df['label'] = 0
    malig_df['label'] = 1
    data = pd.concat([normal_df, malig_df]).reset_index(drop=True)
    y = data.pop('label').values
    if scaler:
        x = scaler.transform(data)
    else:
        scaler = RobustScaler()
        x = scaler.fit_transform(data)
    return x, y, scaler

# ✅ 개별 train_case 학습 및 평가 함수 (LGBM + MLP 적용)
def process_train_case(train_case):
    case_path = os.path.join(BASE_PATH, train_case)
    print(f"\n🔹 Processing {train_case} ...")

    # 🔹 데이터 로드
    train_normal = load_csv(os.path.join(case_path, 'train_Normal.csv'))
    train_malig = load_csv(os.path.join(case_path, 'train_Malig.csv'))
    if train_normal is None or train_malig is None:
        return []

    # 🔹 데이터 전처리
    train_x, train_y, scaler = prepare_data(train_normal, train_malig)

    # ✅ 결과 저장 폴더 생성
    model_save_dir = os.path.join(RESULTS_PATH, train_case)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # ✅ LGBM 학습
    lgbm_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        random_state=42
    )
    lgbm_model.fit(train_x, train_y)
    
    lgbm_model_path = os.path.join(model_save_dir, f"{train_case}_lgbm_model.txt")
    lgbm_model.booster_.save_model(lgbm_model_path)
    print(f"✅ LGBM 모델 저장 완료: {lgbm_model_path}")
    
    # ✅ MLP 학습
    mlp_model = Sequential([
        Dense(256, activation='relu', input_shape=(train_x.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    mlp_model.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    mlp_model_path = os.path.join(model_save_dir, f"{train_case}_mlp_model.h5")
    mlp_model.save(mlp_model_path)
    print(f"✅ MLP 모델 저장 완료: {mlp_model_path}")
    
    # 🔹 평가
    train_pred = (lgbm_model.predict_proba(train_x)[:, 1] > 0.5).astype(int)
    train_acc = accuracy_score(train_y, train_pred)
    train_loss = log_loss(train_y, lgbm_model.predict_proba(train_x)[:, 1])
    
    val_normal = load_csv(os.path.join(case_path, 'val_internal_Normal.csv'))
    val_malig = load_csv(os.path.join(case_path, 'val_internal_Malig.csv'))
    if val_normal is not None and val_malig is not None:
        val_x, val_y, _ = prepare_data(val_normal, val_malig, scaler)
        val_pred = (lgbm_model.predict_proba(val_x)[:, 1] > 0.5).astype(int)
        val_acc = accuracy_score(val_y, val_pred)
        val_loss = log_loss(val_y, lgbm_model.predict_proba(val_x)[:, 1])
    else:
        val_acc, val_loss = None, None

    gen_gap = abs(train_acc - val_acc) if val_acc is not None else None
    loss_ratio = train_loss / val_loss if val_loss is not None and val_loss != 0 else None

    # ✅ 결과 저장
    results = [[train_case, "Gene set", f1_score(train_y, train_pred), recall_score(train_y, train_pred),
                precision_score(train_y, train_pred), accuracy_score(train_y, train_pred),
                train_acc, val_acc, gen_gap, loss_ratio]]
    results_df = pd.DataFrame(results, columns=["Train Case", "Gene set", "F1 Score", "Recall", "Precision", "Accuracy", "Train Acc", "Val Acc", "Generalization Gap", "Loss Ratio"])
    results_csv_path = os.path.join(model_save_dir, "training_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ 학습 결과 저장 완료: {results_csv_path}")

# ✅ 학습 실행
train_cases = sorted([d for d in os.listdir(BASE_PATH) if d.startswith("train_case_")])
for case in train_cases:
    process_train_case(case)

print("\n✅ 모든 LGBM + MLP 학습이 완료되었습니다!")





# ✅ 경로 설정
RESULTS_PATH = "/content/drive/MyDrive/hepscope/results/lgbm_and_mlp_5samples/"
MERGED_RESULTS_PATH = os.path.join(RESULTS_PATH, "merged_lgbm_and_mlp_5samples_training_results.csv")

# ✅ 병합할 파일 리스트 가져오기
train_cases = sorted([d for d in os.listdir(RESULTS_PATH) if d.startswith("train_case_")])
all_results = []

# ✅ 각 train_case의 결과 병합
for case in train_cases:
    result_file = os.path.join(RESULTS_PATH, case, "training_results.csv")
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        all_results.append(df)
        print(f"✅ {case} 결과 추가 완료")
    else:
        print(f"❌ {case} 결과 파일 없음: {result_file}")

# ✅ 병합된 결과 저장
if all_results:
    merged_df = pd.concat(all_results, ignore_index=True)
    merged_df.to_csv(MERGED_RESULTS_PATH, index=False)
    print(f"✅ 모든 학습 결과 병합 완료: {MERGED_RESULTS_PATH}")
else:
    print("❌ 병합할 학습 결과 없음")
