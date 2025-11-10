# Preprocessing
preprocess_heatmaps:
	python preprocess_heatmaps.py

fix_csv_paths:
	python fix_csv_paths.py

# Setup
split:
	python create_splits.py --config-path configs/data_egd-cxr.yaml \
		--output-dir configs/splits --folds 5 --test 0.1 --seed 42 --print-summary

sanity_check:
	python sanity_check_dataset.py --config configs/data_egd-cxr.yaml --batch-size 8 --num-workers 0 --max-fixations 64 --out-dir sample

# Training
train_gnn:
	python main_train_gnn.py --config configs/data_egd-cxr.yaml

train_unet:
	python main_train_unet.py --config configs/train_unet.yaml

train_temporal:
	python main_train_temporal.py --config configs/train_temporal.yaml

train_gradia:
	python main_train_gradia.py --config configs/data_egd-cxr.yaml --attention_weight 0.0 --iou_samples 50

# Evaluation
eval_unet:
	python evaluate_unet.py

eval_temporal:
	python evaluate_temporal.py

eval_gradia:
	python main_train_gradia.py --config configs/data_egd-cxr.yaml --evaluate --checkpoint runs/checkpoints/gradia_best.pt --iou_samples 50
