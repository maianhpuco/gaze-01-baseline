PYTHON ?= python

# .PHONY: prepare_transcripts test_dataset train_model eval_model

split:
	python create_splits.py --config-path configs/data_egd-cxr.yaml \
	      --output-dir configs/splits --folds 5 --test 0.1 --seed 42 --print-summary 

sanity_check_dataset:
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample

sanity_check_dataset_python:
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample --split train
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample --split val
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample --split test
