## Import libraries
from google.colab import drive
drive.mount('/content/drive')

import matplotlib as mpl
mpl.style.use('seaborn-v0_8-whitegrid')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.utils.class_weight import compute_class_weight

## Import data
### Train - Normal : 4000, Malig : 4000
train_aml_Malig = pd.read_csv('/content/drive/MyDrive/hepscope/AML/train_Malig.csv').drop('Unnamed: 0', axis=1)
train_aml_Normal = pd.read_csv('/content/drive/MyDrive/hepscope/AML/train_Normal.csv').drop('Unnamed: 0', axis=1)

train_aml_Normal['label'] = 0
train_aml_Malig['label'] = 1
train_aml_X = pd.concat([train_aml_Normal, train_aml_Malig]).reset_index(drop=True)
train_aml_Y = train_aml_X['label'].values
train_aml_X = train_aml_X.drop(columns='label')

### Validation - Normal : 1000, Malig : 1000
val_aml_Normal = pd.read_csv("/content/drive/MyDrive/hepscope/AML/val_Normal.csv").drop('Unnamed: 0', axis=1)
val_aml_Malig = pd.read_csv("/content/drive/MyDrive/hepscope/AML/val_Malig.csv").drop('Unnamed: 0', axis=1)

val_aml_Normal['label'] = 0
val_aml_Malig['label'] = 1
val_aml_X = pd.concat([val_aml_Normal, val_aml_Malig]).reset_index(drop=True)
val_aml_Y = val_aml_X['label'].values
val_aml_X = val_aml_X.drop(columns='label')

### Test - Normal : 1000, Malig : 1000
test_aml_Normal = pd.read_csv("/content/drive/MyDrive/hepscope/AML/test_Normal.csv").drop('Unnamed: 0', axis=1)
test_aml_Malig = pd.read_csv("/content/drive/MyDrive/hepscope/AML/test_Malig.csv").drop('Unnamed: 0', axis=1)

test_aml_Normal['label'] = 0
test_aml_Malig['label'] = 1
test_aml_X = pd.concat([test_aml_Normal, test_aml_Malig]).reset_index(drop=True)
test_aml_Y = test_aml_X['label'].values
test_aml_X = test_aml_X.drop(columns='label')

## Print dataset shapes and preview
print("train_aml_X shape:", train_aml_X.shape)
print("val_aml_X shape:", val_aml_X.shape)
print("test_aml_X shape:", test_aml_X.shape)

# 클래스 불균형 해결을 위한 class_weight 계산
class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.unique(train_aml_Y), 
                                     y=train_aml_Y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 새로운 MLP 모델 정의
mlp_model = Sequential([
    Input(shape=(2890,)),  # ✅ 2890차원 데이터 그대로 사용
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(1, activation='sigmoid')  # 이진 분류 (AML 여부)
])

# 모델 컴파일
mlp_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 모델 구조 확인
mlp_model.summary()

# 모델 학습
history = mlp_model.fit(train_aml_X, train_aml_Y,
                        validation_data=(val_aml_X, val_aml_Y),
                        epochs=50,
                        batch_size=64,
                        class_weight=class_weight_dict,  # 클래스 불균형 해결
                        verbose=1)

# 모델 평가
test_pred = mlp_model.predict(test_aml_X, batch_size=64)
res_aml_y = [1 if res > 0.5 else 0 for res in test_pred.flatten()]

# 모델 성능 평가
print("Confusion Matrix:")
print(confusion_matrix(test_aml_Y, res_aml_y))
print("Accuracy:", accuracy_score(test_aml_Y, res_aml_y))
print("AUC:", roc_auc_score(test_aml_Y, test_pred))
print("Precision:", precision_score(test_aml_Y, res_aml_y))
print("Recall:", recall_score(test_aml_Y, res_aml_y))
print("F1 Score:", f1_score(test_aml_Y, res_aml_y))
