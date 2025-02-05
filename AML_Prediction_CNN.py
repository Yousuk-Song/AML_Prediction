## Import libraries
from google.colab import drive
drive.mount('/content/drive')

import matplotlib as mpl
mpl.style.use('seaborn-v0_8-whitegrid')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # load_model import 추가

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

## Import pre-trained model
aml_model = load_model('/content/drive/MyDrive/hepscope/Example/hep_model.h5', compile=False)

# 모델 구조 확인 (입력 shape 확인)
aml_model.summary()

# 모델의 입력 shape 확인 후 데이터 shape 조정 필요 여부 체크
expected_input_shape = aml_model.input_shape[1]  # (None, feature_dim)에서 feature_dim만 추출
actual_input_shape = test_aml_X.shape[1]

if expected_input_shape != actual_input_shape:
    print(f"⚠️ 모델이 예상하는 입력 shape: {expected_input_shape}, 제공된 데이터 shape: {actual_input_shape}")
    print("입력 차원을 맞추기 위해 차원 축소(PCA 등) 또는 모델 재학습이 필요합니다.")
else:
    print("✅ 입력 shape가 일치합니다. 예측을 진행합니다.")

    ## Check the performance of model
    aml_pred = aml_model.predict(test_aml_X, batch_size=64)

    ## Convert predictions to binary values
    res_aml_y = [1 if res > 0.5 else 0 for res in aml_pred.flatten()]

    ## Confusion matrix
    print(confusion_matrix(test_aml_Y, res_aml_y))
    print("Accuracy : %.4f " % accuracy_score(test_aml_Y, res_aml_y))

    ## Calculate performance metrics
    aml_fpr, aml_tpr, _ = roc_curve(test_aml_Y, aml_pred)
    aml_roc_auc = auc(aml_fpr, aml_tpr)
    aml_precision, aml_recall, _ = precision_recall_curve(test_aml_Y, aml_pred)
    aml_pr_auc = auc(aml_recall, aml_precision)
    aml_f1 = f1_score(test_aml_Y, res_aml_y)
    aml_Precision = precision_score(test_aml_Y, res_aml_y)
    aml_Recall = recall_score(test_aml_Y, res_aml_y)

    ## AUC, AUPRC, Precision, Recall, F1 score
    print("aml_AUC : %.4f" % aml_roc_auc)
    print("aml_AUPRC : %.4f" % aml_pr_auc)
    print("aml_precision : %.4f" % aml_Precision)
    print("aml_recall : %.4f" % aml_Recall)
    print("aml_f1 : %.4f" % aml_f1)
