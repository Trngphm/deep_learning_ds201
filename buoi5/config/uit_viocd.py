# =====================
# DATA
# =====================
DATA_CONFIG = {
    "train_file": "data/UIT-ViOCD/train.json",
    "dev_file": "data/UIT-ViOCD/dev.json",
    "test_file": "data/UIT-ViOCD/test.json",
    "dataset_name": "uit_viocd",
    "task": "domain_classification",
    "text_field": "review",
    "label_field": "domain",
    "min_freq": 1,
    "max_len": None     
}

# =====================
# MODEL – TRANSFORMER
# =====================
MODEL_CONFIG = {
    "model_name": "TransformerEncoder",
    "d_model": 256,
    "n_head": 8,
    "n_layer": 3,       
    "d_ff": 512,
    "dropout": 0.1,
    "pooling": "mean"  
}

# =====================
# TRAINING
# =====================
TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "grad_clip": 1.0
}

# =====================
# EVALUATION
# =====================
EVAL_CONFIG = {
    "metric": ["accuracy"],
    "save_best": True,
    "monitor": "accuracy"
}

# =====================
# SYSTEM
# =====================
SYSTEM_CONFIG = {
    "device": "cpu",   
    "seed": 42,
    "num_workers": 2
}
