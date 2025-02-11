# ✅ 필요한 패키지 설치 (Colab 환경에서 실행 시 필요)
!pip uninstall -y torchvision
!pip install torchvision --no-cache-dir

!pip install import_ipynb
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install umap-learn
!pip install imbalanced-learn
!pip install xgboost
!pip install scikeras

# ✅ 라이브러리 임포트
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from umap import UMAP
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# ✅ Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# ✅ 데이터 경로 설정
base_path = "/content/drive/MyDrive/hepscope/DN/"

# ✅ 파일 로드 함수
def load_csv(file_name):
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
    else:
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")

# ✅ Train Data
train_aml_Malig = load_csv('train_Malig.csv')
train_aml_Normal = load_csv('train_Normal.csv')

train_aml_Normal['label'] = 0
train_aml_Malig['label'] = 1
train_aml_X = pd.concat([train_aml_Normal, train_aml_Malig]).reset_index(drop=True)
train_aml_Y = train_aml_X['label'].values
train_aml_X = train_aml_X.drop(columns='label')

# ✅ Validation Data (Internal + External)
val_aml_Normal = load_csv("val_internal_Normal.csv")
val_aml_Malig = load_csv("val_internal_Malig.csv")
val_aml_Normal['label'] = 0
val_aml_Malig['label'] = 1
val_aml_X = pd.concat([val_aml_Normal, val_aml_Malig]).reset_index(drop=True)
val_aml_Y = val_aml_X['label'].values
val_aml_X = val_aml_X.drop(columns='label')

external_aml_Normal = load_csv("val_external_Normal.csv")
external_aml_Malig = load_csv("val_external_Malig.csv")
external_aml_Normal['label'] = 0
external_aml_Malig['label'] = 1
external_aml_X = pd.concat([external_aml_Normal, external_aml_Malig]).reset_index(drop=True)
external_aml_Y = external_aml_X['label'].values
external_aml_X = external_aml_X.drop(columns='label')

# ✅ 데이터 정규화 (StandardScaler 적용)
scaler = StandardScaler()
train_aml_X = scaler.fit_transform(train_aml_X)
val_aml_X = scaler.transform(val_aml_X)
external_aml_X = scaler.transform(external_aml_X)

# ✅ SMOTE 적용 (데이터 불균형 해결)
smote = SMOTE(sampling_strategy='auto', random_state=42)
train_aml_X, train_aml_Y = smote.fit_resample(train_aml_X, train_aml_Y)

# ✅ UMAP 적용 (Feature Engineering)
umap = UMAP(n_components=50, random_state=42)
train_aml_X = umap.fit_transform(train_aml_X)
val_aml_X = umap.transform(val_aml_X)
external_aml_X = umap.transform(external_aml_X)

# ✅ Compute Class Weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_aml_Y), y=train_aml_Y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ✅ Define Optimized MLP Model
mlp_model = Sequential([
    Input(shape=(train_aml_X.shape[1],)),

    Dense(512, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.4),

    Dense(256, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.4),

    Dense(128, kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    Dropout(0.4),

    Dense(1, activation='sigmoid')
])

# ✅ Compile Model
mlp_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# ✅ Early Stopping 적용
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Train Model
history = mlp_model.fit(train_aml_X, train_aml_Y,
                        validation_data=(val_aml_X, val_aml_Y),
                        epochs=50,
                        batch_size=32,
                        class_weight=class_weight_dict,
                        callbacks=[early_stopping],
                        verbose=1)

# ✅ Predict
internal_pred = mlp_model.predict(val_aml_X, batch_size=32).flatten()
external_pred = mlp_model.predict(external_aml_X, batch_size=32).flatten()

# ✅ PR Curve 기반 Threshold 최적화
precisions, recalls, thresholds = precision_recall_curve(external_aml_Y, external_pred)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print("🔹 Optimal Threshold:", optimal_threshold)

# ✅ 최적 Threshold 적용
internal_res = [1 if res > optimal_threshold else 0 for res in internal_pred]
external_res = [1 if res > optimal_threshold else 0 for res in external_pred]

# ✅ Performance Metrics 출력 함수
def print_metrics(y_true, y_pred, name):
    print(f"\n📌 {name} Performance")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

print_metrics(val_aml_Y, internal_res, "Internal Validation")
print_metrics(external_aml_Y, external_res, "External Validation (Optimized Threshold)")

# ✅ Confusion Matrix 시각화
def plot_confusion_matrix(y_true, y_pred, title, cmap):
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap=cmap,
                xticklabels=["Normal", "Malig"], yticklabels=["Normal", "Malig"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion_matrix(val_aml_Y, internal_res, "Confusion Matrix (Internal Validation)", "Blues")
plot_confusion_matrix(external_aml_Y, external_res, "Confusion Matrix (External Validation)", "Oranges")
