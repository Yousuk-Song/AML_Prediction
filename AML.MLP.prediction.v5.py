# ✅ 필수 패키지 설치 (Colab 실행 시 필요)
!pip install import_ipynb umap-learn imbalanced-learn xgboost scikeras joblib

# ✅ 라이브러리 임포트
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, GaussianNoise
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from google.colab import drive
from joblib import Parallel, delayed

# ✅ Google Drive 마운트
drive.mount('/content/drive')

# ✅ 데이터 경로 설정
BASE_PATH = "/content/drive/MyDrive/hepscope/train_val_sample_test/"
RESULTS_PATH = "/content/drive/MyDrive/hepscope/results/"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ✅ Confusion Matrix 시각화 함수
def plot_confusion_matrix(y_true, y_pred, train_case, validation_name):
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap="Oranges",
                xticklabels=["Normal", "Malig"], yticklabels=["Normal", "Malig"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix: {train_case} ({validation_name})")
    save_path = os.path.join(RESULTS_PATH, f"Confusion_Matrix_{train_case}_{validation_name}.png")
    plt.savefig(save_path)
    plt.close()

# ✅ Loss Curve 시각화 함수
def plot_loss_curve(history, train_case, validation_name):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve: {train_case} ({validation_name})")
    plt.legend()
    save_path = os.path.join(RESULTS_PATH, f"Loss_Curve_{train_case}_{validation_name}.png")
    plt.savefig(save_path)
    plt.close()

# ✅ 파일 로드 함수
def load_csv(file_path):
    try:
        return pd.read_csv(file_path).drop(columns=['Unnamed: 0'], errors='ignore')
    except Exception as e:
        print(f"❌ 파일 로드 실패 ({file_path}): {e}")
        return None

# ✅ MLP 모델 생성 함수 (더 깊고 Dropout 증가)
def get_mlp_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        GaussianNoise(0.05),  # ✅ Noise Injection
        Dense(256, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ 데이터 정규화 및 결합 함수 (RobustScaler 사용)
def prepare_data(normal_df, malig_df, scaler=None):
    if normal_df is None or malig_df is None:
        return None, None

    normal_df['label'] = 0
    malig_df['label'] = 1
    data = pd.concat([normal_df, malig_df]).reset_index(drop=True)
    y = data.pop('label').values
    
    if scaler:
        x = scaler.transform(data)
    else:
        scaler = RobustScaler()  # ✅ Outlier에 강한 Scaling 적용
        x = scaler.fit_transform(data)
    
    return x, y, scaler

# ✅ 개별 train_case 학습 및 평가 함수
def process_train_case(train_case):
    case_path = os.path.join(BASE_PATH, train_case)
    print(f"\n🔹 Processing {train_case} ...")

    # ✅ Train 데이터 로드
    train_normal = load_csv(os.path.join(case_path, 'train_Normal.csv'))
    train_malig = load_csv(os.path.join(case_path, 'train_Malig.csv'))

    if train_normal is None or train_malig is None:
        print(f"⚠️ {train_case} 데이터 없음. 스킵.")
        return []

    train_x, train_y, scaler = prepare_data(train_normal, train_malig)

    # ✅ 모델 생성
    model = get_mlp_model(input_dim=train_x.shape[1])

    # ✅ Learning Rate Scheduling & Early Stopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # ✅ External Validation 데이터 폴더 확인
    external_folders = [d for d in os.listdir(case_path) if d.startswith("val_") and os.path.isdir(os.path.join(case_path, d))]
    results = []

    for external_case in external_folders:
        external_path = os.path.join(case_path, external_case)
        external_normal = load_csv(os.path.join(external_path, "val_external_Normal.csv"))
        external_malig = load_csv(os.path.join(external_path, "val_external_Malig.csv"))

        if external_normal is not None and external_malig is not None:
            external_x, external_y, _ = prepare_data(external_normal, external_malig, scaler)

            # ✅ 모델 학습
            history = model.fit(train_x, train_y, validation_data=(external_x, external_y),
                                epochs=30, batch_size=64,
                                callbacks=[reduce_lr, early_stopping], verbose=1)

            # ✅ 결과 저장
            plot_loss_curve(history, train_case, external_case)
            pred = (model.predict(external_x) > 0.5).astype(int)
            plot_confusion_matrix(external_y, pred, train_case, external_case)

            results.append([train_case, external_case, accuracy_score(external_y, pred),
                            precision_score(external_y, pred), recall_score(external_y, pred), f1_score(external_y, pred)])

    return results
# ✅ 병렬 처리 실행
train_cases = sorted([d for d in os.listdir(BASE_PATH) if d.startswith("train_case_")])
all_results = Parallel(n_jobs=-1)(delayed(process_train_case)(case) for case in train_cases)

# ✅ 결과 저장
results_df = pd.DataFrame([row for case_result in all_results for row in case_result],
                          columns=["Train Case", "External Case", "Accuracy", "Precision", "Recall", "F1 Score"])
results_csv_path = os.path.join(RESULTS_PATH, "External_Validation_Results.csv")
results_df.to_csv(results_csv_path, index=False)

print(f"\n✅ 모든 External Validation 결과가 {results_csv_path}에 저장되었습니다!")
