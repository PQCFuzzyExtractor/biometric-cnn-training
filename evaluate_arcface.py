# evaluate_arcface.py
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CASIAIrisDataset
from model import IrisNet  # ArcFace 학습 때 썼던 백본 그대로


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    ])


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a ^ b).sum().item())


def main():
    # 1) Dataset
    transform = build_transform()
    dataset = CASIAIrisDataset("data/CASIA", transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 2) ArcFace 백본 로드
    model = IrisNet(out_dim=3488).to(DEVICE)
    state_dict = torch.load("checkpoints/arcface_backbone.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Loaded ArcFace backbone from checkpoints/arcface_backbone.pth")

    # 3) 전체 임베딩 수집
    all_embs = []
    all_pids = []

    with torch.no_grad():
        for imgs, pids in loader:
            imgs = imgs.to(DEVICE)
            emb = model(imgs)           # (B,3488)
            all_embs.append(emb.cpu())
            all_pids.extend(pids.tolist())

    embs = torch.cat(all_embs, dim=0)      # (N,3488)
    pids_tensor = torch.tensor(all_pids)   # (N,)

    print(f"[INFO] Total samples: {embs.size(0)}")

    # 4) 차원별 threshold = 중앙값 (median)
    #    → 각 비트가 0/1로 균형 잡히게
    thresholds = embs.median(dim=0).values  # (3488,)

    # 5) 이진화
    binary_codes = (embs > thresholds).to(torch.uint8)  # (N,3488)

    # 6) PID별 인덱스 맵
    pid_to_indices: Dict[int, List[int]] = {}
    for idx, pid in enumerate(all_pids):
        pid_to_indices.setdefault(pid, []).append(idx)

    same_candidates = [pid for pid, idxs in pid_to_indices.items() if len(idxs) >= 2]
    diff_candidates = list(pid_to_indices.keys())

    same_hd = []
    diff_hd = []
    num_pairs = 2000

    # Same-person
    for _ in range(num_pairs):
        pid = random.choice(same_candidates)
        idxs = pid_to_indices[pid]
        i, j = random.sample(idxs, 2)
        hd = hamming_distance(binary_codes[i], binary_codes[j])
        same_hd.append(hd)

    # Diff-person
    for _ in range(num_pairs):
        pid1, pid2 = random.sample(diff_candidates, 2)
        i = random.choice(pid_to_indices[pid1])
        j = random.choice(pid_to_indices[pid2])
        hd = hamming_distance(binary_codes[i], binary_codes[j])
        diff_hd.append(hd)

    same_avg = sum(same_hd) / len(same_hd)
    diff_avg = sum(diff_hd) / len(diff_hd)

    print("---- ArcFace Evaluation Result ----")
    print(f"Same person HD avg : {same_avg:.2f}")
    print(f"Diff person HD avg : {diff_avg:.2f}")
    print(f"(num_pairs={num_pairs})")


if __name__ == "__main__":
    main()
