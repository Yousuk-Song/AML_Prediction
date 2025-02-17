from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import pandas as pd
import numpy as np
from google.colab import drive

# ✅ Google Drive 마운트
drive.mount('/content/drive')

# ✅ 모델 로드
model_path = "/content/drive/MyDrive/hepscope/results/train_case_22/train_case_22_model.h5"
model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

# ✅ 새로운 데이터 로드
#new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/train_case_22/val_GSE185381/val_external_Malig.csv"
#new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/train_case_22/val_GSE185381/val_external_Normal.csv"
#new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/train_case_22/train_Malig.csv"
#new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/train_case_22/train_Normal.csv"
#new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/train_case_22/val_internal_Malig.csv"
#new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/train_case_22/val_internal_Normal.csv"
new_data_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/pt01_bm01.norm.csv"
new_data = pd.read_csv(new_data_path, index_col=0)  # 첫 번째 컬럼을 인덱스로 설정 (cell_id)


# ✅ 예측 수행 (추가 정규화 없음)
predictions = model.predict(new_data)
binary_predictions = (predictions > 0.5).astype(int)

# ✅ 결과 DataFrame 생성
result_df = pd.DataFrame({'cell_id': new_data.index, 'Prediction': binary_predictions.flatten()})

# ✅ 0과 1의 개수 및 비율 계산
num_zeros = (result_df['Prediction'] == 0).sum()
num_ones = (result_df['Prediction'] == 1).sum()
total = len(result_df)

zero_percent = (num_zeros / total) * 100
one_percent = (num_ones / total) * 100

# ✅ CSV 저장
result_csv_path = "/content/drive/MyDrive/hepscope/Train_case_22_5sample/new_prediction.csv"
result_df.to_csv(result_csv_path, index=False, sep="\t")  # TSV 형식으로 저장 (탭 구분자)

# ✅ 결과 출력
print(f"✅ 예측 결과가 {result_csv_path} 에 저장되었습니다!")
print(f"0의 개수: {num_zeros} ({zero_percent:.2f}%)")
print(f"1의 개수: {num_ones} ({one_percent:.2f}%)")
