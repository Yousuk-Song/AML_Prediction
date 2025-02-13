## Import necessary libraries
library(Seurat)
library(data.table)
library(Matrix)
library(dplyr)

setwd("/home/song7602/1.scripts/HepScope")

## Load Seurat RDS file
aml.test <- readRDS("/data/workbench/scRSEQ_AML/data/aml.cfinder.QC.rds")
aml.test <- JoinLayers(aml.test)

## Set Default Assay to RNA (Prevents AB assay issues)
DefaultAssay(aml.test) <- "RNA"

## Load Gene List
valid_genes <- fread("/home/song7602/1.scripts/HepScope/aml.cancer.marker.intersect_gene.txt", header = FALSE)$V1
valid_genes <- valid_genes[valid_genes %in% rownames(aml.test)]

aml.subset <- subset(aml.test, features = valid_genes)
aml.subset <- NormalizeData(aml.subset)

## Ensure 'label' is present in metadata
if (!("label" %in% colnames(aml.subset@meta.data))) {
  stop("Error: 'label' column is missing from meta.data. Check your input data.")
}

## Train/Validation Split 조합 데이터 불러오기
split_data <- fread("/home/song7602/1.scripts/HepScope/train_val_split.txt", header = TRUE)

## 샘플 수 고정 (무조건 4000, 1000, 1000 샘플링)
total_train_samples <- 4000
total_val_internal_samples <- 1000
total_val_external_samples <- 1000

## 각 train 조합별 반복 실행
for (i in 1:nrow(split_data)) {
  train_tag <- colnames(split_data)[which(split_data[i, ] == "TRAIN")]
  valid_tags <- colnames(split_data)[which(split_data[i, ] == "VALID")]
  
  print(paste("\n🔹 Processing Train Case:", i))
  print(paste("🔹 Train:", train_tag))
  print(paste("🔹 Validation Sets:", valid_tags))
  
  ## Train 디렉토리 생성
  train_dir <- paste0("train_case_", i)
  dir.create(train_dir, showWarnings = FALSE)
  
  ## Train 데이터 필터링
  train_cells <- Cells(aml.subset)[aml.subset@meta.data$sample_tag %in% train_tag]
  
  if (length(train_cells) == 0) {
    print(paste("⚠️ Warning: No cells found in the train dataset for", train_tag, ". Skipping this case."))
    next
  }
  
  aml_train_all <- subset(aml.subset, cells = train_cells)
  
  ## 사용 가능한 Normal, Malignant 개수 확인
  available_normal <- sum(aml_train_all@meta.data$label == "0")
  available_malig <- sum(aml_train_all@meta.data$label == "1")
  
  print(paste("✅ Available Normal:", available_normal, "| Available Malignant:", available_malig))
  
  ## 총 세포 수가 부족하면 스킵
  if ((available_normal + available_malig) < (total_train_samples + total_val_internal_samples)) {
    print(paste("⚠️ Warning: Not enough total cells for", train_tag, ". Skipping this case."))
    next
  }
  
  ## Train 샘플링
  train_normal_cells <- sample(Cells(aml_train_all)[aml_train_all@meta.data$label == "0"], total_train_samples / 2)
  train_malig_cells <- sample(Cells(aml_train_all)[aml_train_all@meta.data$label == "1"], total_train_samples / 2)
  
  aml_train_normal <- subset(aml_train_all, cells = train_normal_cells)
  aml_train_malig <- subset(aml_train_all, cells = train_malig_cells)
  
  ## Internal Validation (Test) 샘플링
  remaining_normal_cells <- setdiff(Cells(aml_train_all)[aml_train_all@meta.data$label == "0"], train_normal_cells)
  remaining_malig_cells <- setdiff(Cells(aml_train_all)[aml_train_all@meta.data$label == "1"], train_malig_cells)
  
  internal_normal_cells <- sample(remaining_normal_cells, total_val_internal_samples / 2)
  internal_malig_cells <- sample(remaining_malig_cells, total_val_internal_samples / 2)
  
  aml_val_internal_normal <- subset(aml_train_all, cells = internal_normal_cells)
  aml_val_internal_malig <- subset(aml_train_all, cells = internal_malig_cells)
  
  ## CSV 저장 (Train & Internal Val)
  write.csv(as.data.frame(t(as.matrix(GetAssayData(aml_train_normal, slot = "counts")))), 
            file.path(train_dir, "train_Normal.csv"), row.names = TRUE)
  write.csv(as.data.frame(t(as.matrix(GetAssayData(aml_train_malig, slot = "counts")))), 
            file.path(train_dir, "train_Malig.csv"), row.names = TRUE)
  
  write.csv(as.data.frame(t(as.matrix(GetAssayData(aml_val_internal_normal, slot = "counts")))), 
            file.path(train_dir, "val_internal_Normal.csv"), row.names = TRUE)
  write.csv(as.data.frame(t(as.matrix(GetAssayData(aml_val_internal_malig, slot = "counts")))), 
            file.path(train_dir, "val_internal_Malig.csv"), row.names = TRUE)
  
  ## External Validation Set 처리
  for (val in valid_tags) {
    val_dir <- file.path(train_dir, paste0("val_", val))
    dir.create(val_dir, showWarnings = FALSE)
    
    val_cells <- Cells(aml.subset)[aml.subset@meta.data$sample_tag == val]
    
    if (length(val_cells) == 0) {
      print(paste("⚠️ Warning: No cells found for External Validation Set:", val, "Skipping this set."))
      next
    }
    
    aml_val_external <- subset(aml.subset, cells = val_cells)
    
    val_external_available_normal <- sum(aml_val_external@meta.data$label == "0")
    val_external_available_malig <- sum(aml_val_external@meta.data$label == "1")
    
    if (val_external_available_normal < total_val_external_samples / 2 | val_external_available_malig < total_val_external_samples / 2) {
      print(paste("⚠️ Warning: Not enough cells for external validation in", val, ". Skipping this set."))
      next
    }
    
    external_normal_cells <- sample(Cells(aml_val_external)[aml_val_external@meta.data$label == "0"], total_val_external_samples / 2)
    external_malig_cells <- sample(Cells(aml_val_external)[aml_val_external@meta.data$label == "1"], total_val_external_samples / 2)
    
    aml_val_external_normal <- subset(aml_val_external, cells = external_normal_cells)
    aml_val_external_malig <- subset(aml_val_external, cells = external_malig_cells)
    
    ## CSV 저장 (External Val)
    write.csv(as.data.frame(t(as.matrix(GetAssayData(aml_val_external_normal, slot = "counts")))), 
              file.path(val_dir, "val_external_Normal.csv"), row.names = TRUE)
    write.csv(as.data.frame(t(as.matrix(GetAssayData(aml_val_external_malig, slot = "counts")))), 
              file.path(val_dir, "val_external_Malig.csv"), row.names = TRUE)
  }
}


