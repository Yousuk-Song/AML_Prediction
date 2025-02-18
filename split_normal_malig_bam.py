#!/usr/bin/python

import pysam
import pandas as pd
import subprocess
import sys

# 1️⃣ 파일 경로 설정
#bam_file = "pt01.bm01.AML02_with_RGmapping_quality_adjusted.bam"
#csv_file = "pt01_bm01_prediction.csv"
#sample_numb = "pt01_bm01"
bam_file = sys.argv[1]
csv_file = sys.argv[2]
sample_numb = sys.argv[3]
normal_bam = bam_file.replace(".bam", ".normal.bam")
malig_bam = bam_file.replace(".bam", ".malig.bam")

# 2️⃣ CSV 파일 읽기
df = pd.read_csv(csv_file, sep="\t")  # TSV 파일이므로 tab-separated로 읽기
df["cell_id"] = df["cell_id"].str.replace(sample_numb + "_", "", regex=False)  # "pt01_bm01_" 제거

# 3️⃣ Normal (0)과 Malignant (1) CB 리스트 생성
normal_cb = set(df[df["Prediction"] == 0]["cell_id"].astype(str))  # set()으로 변환하여 빠른 검색
malig_cb = set(df[df["Prediction"] == 1]["cell_id"].astype(str))

# 4️⃣ 원본 BAM 파일 열기
bam_in = pysam.AlignmentFile(bam_file, "rb")  # 입력 BAM 파일 (읽기 전용)

# 5️⃣ 새로운 BAM 파일 생성 (헤더 포함)
bam_normal_out = pysam.AlignmentFile(normal_bam, "wb", header=bam_in.header)
bam_malig_out = pysam.AlignmentFile(malig_bam, "wb", header=bam_in.header)

# 6️⃣ BAM 파일 읽고 CB 태그를 기준으로 분할
for read in bam_in:
    tags = dict(read.tags)  # 태그를 딕셔너리로 변환
    cb = tags.get("CB")  # Cell Barcode (CB) 태그 가져오기

    if cb:
        if cb in normal_cb:
            bam_normal_out.write(read)  # Normal BAM에 저장
        elif cb in malig_cb:
            bam_malig_out.write(read)  # Malignant BAM에 저장

# 7️⃣ 파일 닫기
bam_in.close()
bam_normal_out.close()
bam_malig_out.close()

# 8️⃣ BAM 파일 정렬 및 인덱싱 (Samtools 사용)
subprocess.run(f"samtools sort -@ 40 -o {normal_bam.replace('.bam', '_sorted.bam')} {normal_bam}", shell=True)
subprocess.run(f"samtools index {normal_bam.replace('.bam', '_sorted.bam')}", shell=True)

subprocess.run(f"samtools sort -@ 40 -o {malig_bam.replace('.bam', '_sorted.bam')} {malig_bam}", shell=True)
subprocess.run(f"samtools index {malig_bam.replace('.bam', '_sorted.bam')}", shell=True)

print("✅ BAM 파일 분할 완료: Normal → pt01_bm01_normal_sorted.bam, Malignant → pt01_bm01_malig_sorted.bam")
