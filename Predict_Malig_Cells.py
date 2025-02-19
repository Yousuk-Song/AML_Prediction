from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import pandas as pd
import numpy as np
from google.colab import drive

# ✅ Google Drive 마운트
drive.mount('/content/drive')

# ✅ 모델 로드
model_path = "/content/drive/MyDrive/hepscope/results/5sample/train_case_38/train_case_38_model.h5"
model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

# ✅ 새로운 데이터 로드
new_data_path = "/content/drive/MyDrive/hepscope/Train_case_38_5sample/aml.test.pub.pc200.broad.csv"
result_csv_path = "/content/drive/MyDrive/hepscope/Train_case_38_5sample/aml.test.pub.pc200.broad_prediction.csv"

#pt01_bm01.norm.reversed.5sample.csv
#pt01_bm02.norm.reversed.5sample.csv


new_data = pd.read_csv(new_data_path, index_col=0)  # 첫 번째 컬럼을 인덱스로 설정 (cell_id)

# ✅ 입력 차원 확인 및 패딩 적용
expected_dim = 469  # 모델이 기대하는 입력 차원
current_dim = new_data.shape[1]

if current_dim < expected_dim:
    pad_width = expected_dim - current_dim
    new_data_padded = np.pad(new_data, ((0, 0), (0, pad_width)), mode='constant')
    print(f"입력 데이터에 {pad_width}개의 0을 추가하여 {expected_dim} 차원으로 맞춤.")
elif current_dim > expected_dim:
    new_data_padded = new_data.iloc[:, :expected_dim].values  # 초과되는 feature 자르기
    print(f"입력 데이터에서 초과된 {current_dim - expected_dim}개의 feature를 제거하여 {expected_dim} 차원으로 맞춤.")
else:
    new_data_padded = new_data.values  # 변형 없이 사용

# ✅ 예측 수행 (추가 정규화 없음)
predictions = model.predict(new_data_padded)
binary_predictions = (predictions > 0.5).astype(int)

# ✅ 결과 DataFrame 생성
result_df = pd.DataFrame({'cell_id': new_data.index, 'Prediction': binary_predictions.flatten()})

# ✅ 0과 1의 개수 및 비율 계산
num_zeros = (result_df['Prediction'] == 0).sum()
num_ones = (result_df['Prediction'] == 1).sum()
total = len(result_df)

zero_percent = (num_zeros / total) * 100
one_percent = (num_ones / total) * 100

# ✅ CSV 저장 (TSV 형식)
#result_csv_path = "/content/drive/MyDrive/hepscope/Train_case_38_5sample/aml.test.pub.pc200.broad_prediction.csv"
result_df.to_csv(result_csv_path, index=False, sep="\t")  # 탭 구분자 사용

# ✅ 결과 출력
print(f"✅ 예측 결과가 {result_csv_path} 에 저장되었습니다!")
print(f"0의 개수: {num_zeros} ({zero_percent:.2f}%)")
print(f"1의 개수: {num_ones} ({one_percent:.2f}%)")
