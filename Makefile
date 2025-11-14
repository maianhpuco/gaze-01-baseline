PYTHON ?= python
CONFIG ?= configs/train_gazegnn.yaml
GPUS ?= 0,1

.PHONY: split sanity_check_dataset sanity_check_dataset_python train_gazemtl_main train_gradia train_temporal train_unet eval_all

split:
	python create_splits.py --config-path configs/data_egd-cxr.yaml \
	      --output-dir configs/splits --folds 5 --test 0.1 --seed 42 --print-summary 

sanity_check_dataset:
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample

sanity_check_dataset_python:
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample --split train
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample --split val
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample --split test

# GazeMTL Training

train_gazemtl_main:
	python main_train_gazemtl.py --config configs/train_gazemtl.yaml

# GRADIA Training
train_gradia:
	bash scripts/run_gradia.sh

# Temporal Model Training
train_temporal:
	bash scripts/run_temporal.sh

# UNet Model Training
train_unet:
	bash scripts/run_unet.sh

# Evaluation (all models)
eval_all:
	bash scripts/eval_all.sh $(CONFIG) $(GPUS)