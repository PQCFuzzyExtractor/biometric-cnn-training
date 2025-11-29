# utils.py
import os
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")

def load_model(model_class, path: str, device="cpu"):
    model = model_class()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {path}")
    return model

def hamming_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    return int(np.sum(v1 != v2))
