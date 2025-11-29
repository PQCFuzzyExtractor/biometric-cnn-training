# evaluate.py
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CASIAIrisDataset
from model import IrisNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    ])


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    a, b: (D,) uint8 {0,1}
    """
    return int((a ^ b).sum().item())


def main():
    # 1) 데이터셋 & 로더
    transform = build_transform()
    dataset = CASIAIrisDataset("../data/CASIA", transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 2) 모델 로드
    model = IrisNet().to(DEVICE)
    state_dict = torch.load("checkpoints/triplet_model.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Loaded model from checkpoints/triplet_model.pth")

    # 3) 전체 임베딩 + PID 수집
    all_codes: List[torch.Tensor] = []
    all_pids: List[int] = []

    with torch.no_grad():
        for imgs, pids in loader:
            imgs = imgs.to(DEVICE)
            emb = model(imgs)                   # (B,3488), sigmoid(0~1)
            binary = (emb > 0.5).to(torch.uint8).cpu()   # 이진화
            all_codes.append(binary)
            all_pids.extend(pids.tolist())

    codes = torch.cat(all_codes, dim=0)  # (N,3488)
    pids_tensor = torch.tensor(all_pids, dtype=torch.long)  # (N,)

    print(f"[INFO] Total samples: {codes.size(0)}")

    # 4) PID별 인덱스 맵핑
    pid_to_indices: Dict[int, List[int]] = {}
    for idx, pid in enumerate(all_pids):
        pid_to_indices.setdefault(pid, []).append(idx)

    # 5) Same-person / Diff-person 페어 샘플링
    same_hd_list: List[int] = []
    diff_hd_list: List[int] = []

    # same-person: 최소 2장 이상 가진 PID만
    same_candidates = [pid for pid, inds in pid_to_indices.items() if len(inds) >= 2]
    diff_candidates = list(pid_to_indices.keys())

    num_pairs = 2000  # 샘플링 횟수

    # Same-person pairs
    for _ in range(num_pairs):
        pid = random.choice(same_candidates)
        inds = pid_to_indices[pid]
        i, j = random.sample(inds, 2)

        hd = hamming_distance(codes[i], codes[j])
        same_hd_list.append(hd)

    # Diff-person pairs
    for _ in range(num_pairs):
        pid1, pid2 = random.sample(diff_candidates, 2)
        i = random.choice(pid_to_indices[pid1])
        j = random.choice(pid_to_indices[pid2])

        hd = hamming_distance(codes[i], codes[j])
        diff_hd_list.append(hd)

    same_avg = sum(same_hd_list) / len(same_hd_list)
    diff_avg = sum(diff_hd_list) / len(diff_hd_list)

    print("---- Evaluation Result ----")
    print(f"Same person HD avg : {same_avg:.2f}")
    print(f"Diff person HD avg : {diff_avg:.2f}")
    print(f"(num_pairs={num_pairs})")


if __name__ == "__main__":
    main()
