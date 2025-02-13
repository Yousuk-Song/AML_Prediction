# ✅ 필요한 패키지 설치 (Colab 환경에서 실행 시 필요)
!pip install import_ipynb umap-learn imbalanced-learn xgboost scikeras

# ✅ 라이브러리 임포트
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from google.colab import drive

# ✅ Google Drive 마운트
drive.mount('/content/drive')

# ✅ 데이터 경로 설정
base_path = "/content/drive/MyDrive/hepscope/train_val_sample_test/"
results_path = "/content/drive/MyDrive/hepscope/results/"
os.makedirs(results_path, exist_ok=True)

# ✅ 파일 로드 함수
def load_csv(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
    else:
        print(f"❌ 파일 없음 또는 비어 있음: {file_path}")
        return None

# ✅ Confusion Matrix 시각화 함수
def plot_confusion_matrix(y_true, y_pred, train_case, validation_name, title, cmap):
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap=cmap,
                xticklabels=["Normal", "Malig"], yticklabels=["Normal", "Malig"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{title}: {train_case} ({validation_name})")
    save_path = os.path.join(results_path, f"Confusion_Matrix_{train_case}_{validation_name}.png")
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix 저장됨: {save_path}")
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
    save_path = os.path.join(results_path, f"Loss_Curve_{train_case}_{validation_name}.png")
    plt.savefig(save_path)
    print(f"✅ Loss Curve 저장됨: {save_path}")
    plt.close()

# ✅ train_case 리스트 가져오기
train_cases = sorted([d for d in os.listdir(base_path) if d.startswith("train_case_")])

# ✅ 성능 결과 저장을 위한 리스트
results_list = []

# ✅ 각 train_case별 학습 및 평가
for train_case in train_cases:
    print(f"\n🔹 Processing {train_case} ...")
    case_path = os.path.join(base_path, train_case)

    # ✅ Train Data 불러오기
    train_aml_Malig = load_csv(os.path.join(case_path, 'train_Malig.csv'))
    train_aml_Normal = load_csv(os.path.join(case_path, 'train_Normal.csv'))

    if train_aml_Malig is None or train_aml_Normal is None:
        print(f"⚠️ {train_case} 데이터 없음. 스킵.")
        continue

    train_aml_Normal['label'] = 0
    train_aml_Malig['label'] = 1
    train_aml_X = pd.concat([train_aml_Normal, train_aml_Malig]).reset_index(drop=True)
    train_aml_Y = train_aml_X['label'].values
    train_aml_X = train_aml_X.drop(columns='label')

    # ✅ 데이터 정규화
    scaler = StandardScaler()
    train_aml_X = scaler.fit_transform(train_aml_X)

    # ✅ 과적합 방지된 MLP 모델 정의
    mlp_model = Sequential([
        Input(shape=(train_aml_X.shape[1],)),

        Dense(512, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Dropout(0.4),

        Dense(256, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Dropout(0.4),

        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Dropout(0.4),

        Dense(1, activation='sigmoid')
    ])

    mlp_model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    # ✅ Learning Rate Scheduling & Early Stopping 적용
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    # ✅ External Validation 수행
    external_folders = [
        d for d in os.listdir(case_path)
        if d.startswith("val_") and not d.endswith(".csv") and os.path.isdir(os.path.join(case_path, d))
    ]

    for external_case in external_folders:
        external_path = os.path.join(case_path, external_case)
        external_Normal = load_csv(os.path.join(external_path, "val_external_Normal.csv"))
        external_Malig = load_csv(os.path.join(external_path, "val_external_Malig.csv"))

        if external_Normal is not None and external_Malig is not None:
            external_Normal['label'] = 0
            external_Malig['label'] = 1
            external_X = pd.concat([external_Normal, external_Malig]).reset_index(drop=True)
            external_Y = external_X['label'].values
            external_X = scaler.transform(external_X.drop(columns='label'))

            # ✅ 모델 학습
            history = mlp_model.fit(train_aml_X, train_aml_Y,
                                    validation_data=(external_X, external_Y),
                                    epochs=20, batch_size=32,
                                    callbacks=[reduce_lr, early_stopping],
                                    verbose=1)

            # ✅ Loss Curve 저장
            plot_loss_curve(history, train_case, external_case)

            # ✅ 모델 예측
            external_pred = (mlp_model.predict(external_X) > 0.5).astype(int)

            # ✅ Confusion Matrix 저장
            plot_confusion_matrix(external_Y, external_pred, train_case, external_case, "External Validation", "Oranges")

            # ✅ 성능 평가
            accuracy = accuracy_score(external_Y, external_pred)
            precision = precision_score(external_Y, external_pred)
            recall = recall_score(external_Y, external_pred)
            f1 = f1_score(external_Y, external_pred)

            results_list.append([train_case, external_case, accuracy, precision, recall, f1])

            print(f"\n📌 External Validation Results ({train_case} - {external_case})")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

# ✅ 결과를 DataFrame으로 저장
results_df = pd.DataFrame(results_list, columns=["Train Case", "External Case", "Accuracy", "Precision", "Recall", "F1 Score"])
results_csv_path = os.path.join(results_path, "External_Validation_Results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n✅ 모든 External Validation 결과가 {results_csv_path}에 저장되었습니다!")
