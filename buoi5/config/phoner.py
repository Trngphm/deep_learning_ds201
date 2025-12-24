DATA_CONFIG = {
    "train_file": "data/phoner/train_word.json",
    "dev_file": "data/phoner/dev_word.json",
    "test_file": "data/phoner/test_word.json",
    "dataset_name": "phoner"
}

MODEL_CONFIG = {
    "d_model": 256,
    "n_head": 8,
    "n_layer": 3,
    "d_ff": 512,
    "dropout": 0.1
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4
}

SYSTEM_CONFIG = {
    "device": "cuda",
    "seed": 42
}
