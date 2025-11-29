# train_triplet.py
import os
import random
from typing import List, Tuple

import torch
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from dataset import CASIAIrisDataset
from model import IrisNet
from losses import TripletLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)


# -------------------------
# 공통 Transform (훈련용)
# -------------------------
def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                                # (1,H,W) 0~1
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),     # (3,H,W)로 확장
    ])


# -------------------------
# Triplet 샘플러
#  - dataset.pid_to_imgs 를 사용해서
#    (같은 사람 2장 + 다른 사람 1장)을 batch_size개 뽑는다.
# -------------------------
def generate_triplets(dataset: CASIAIrisDataset, batch_size: int):
    # 이미지가 2장 이상 있는 사람만 anchor/positive 후보
    valid_pids = [pid for pid, imgs in dataset.pid_to_imgs.items() if len(imgs) >= 2]
    if not valid_pids:
        raise RuntimeError("2장 이상 이미지가 있는 PID가 없습니다.")

    all_pids = list(dataset.pid_to_imgs.keys())

    while True:
        anchors: List[str] = []
        positives: List[str] = []
        negatives: List[str] = []

        while len(anchors) < batch_size:
            pid = random.choice(valid_pids)
            imgs = dataset.pid_to_imgs[pid]

            # 같은 사람 이미지 두 장 (anchor, positive)
            anc_path, pos_path = random.sample(imgs, 2)

            # 다른 사람에서 negative 1장
            neg_pid = random.choice(all_pids)
            # 같은 사람/이미지 수 0인 경우는 다시 뽑기
            while neg_pid == pid or len(dataset.pid_to_imgs[neg_pid]) == 0:
                neg_pid = random.choice(all_pids)
            neg_path = random.choice(dataset.pid_to_imgs[neg_pid])

            anchors.append(anc_path)
            positives.append(pos_path)
            negatives.append(neg_path)

        yield anchors, positives, negatives


# -------------------------
# 이미지 경로 리스트 → 텐서 배치
# -------------------------
def load_batch(image_paths: List[str], transform) -> Tensor:
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("L")   # 원본은 그레이스케일
        img = transform(img)
        imgs.append(img)
    return torch.stack(imgs, dim=0)       # (B,3,224,224)


# -------------------------
# 학습 메인
# -------------------------
def train():
    transform = build_transform()

    # dataset.py 에서 transform은 안 쓰고, 여기서 직접 이미지를 연다
    dataset = CASIAIrisDataset("../data/CASIA", transform=None)
    print("[INFO] Persons:", len(dataset.pid_to_imgs))

    batch_size = 32
    epochs = 20

    model = IrisNet().to(DEVICE)
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    sampler = generate_triplets(dataset, batch_size)

    # 너무 오래 돌지 않도록 한 epoch당 step 수를 제한
    steps_per_epoch = max(1, (len(dataset) // batch_size) // 5)

    os.makedirs("../checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}")
        for _ in pbar:
            anc_paths, pos_paths, neg_paths = next(sampler)

            anc = load_batch(anc_paths, transform).to(DEVICE)
            pos = load_batch(pos_paths, transform).to(DEVICE)
            neg = load_batch(neg_paths, transform).to(DEVICE)

            out_a = model(anc)
            out_p = model(pos)
            out_n = model(neg)

            loss = criterion(out_a, out_p, out_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / steps_per_epoch
        print(f"[Epoch {epoch}] Avg Loss = {avg_loss:.4f}")

        ckpt_path = "../checkpoints/triplet_model.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] Saved model to {ckpt_path}")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    train()
