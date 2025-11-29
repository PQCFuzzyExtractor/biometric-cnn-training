# train_arcface.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CASIAIrisDataset
from model import IrisNet
from arcface_head import ArcMarginProduct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # 1채널 → 3채널
    ])


def train_arcface():
    transform = build_transform()
    dataset = CASIAIrisDataset("data/CASIA", transform=transform)

    num_classes = len(dataset.pid_to_imgs)
    print(f"[INFO] Persons (classes): {num_classes}")
    print(f"[INFO] Total samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # 1) Backbone: IrisNet(3488-dim embedding)
    backbone = IrisNet(out_dim=3488).to(DEVICE)

    # 2) ArcFace 헤드
    arc_head = ArcMarginProduct(in_features=3488,
                                out_features=num_classes,
                                s=32.0,
                                m=0.5).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Backbone + Head 함께 최적화
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(arc_head.parameters()),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    epochs = 20
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        backbone.train()
        arc_head.train()

        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"[ArcFace] Epoch {epoch}/{epochs}")
        for imgs, pids in pbar:
            imgs = imgs.to(DEVICE)
            labels = pids.to(DEVICE)

            # 1) embedding 추출
            emb = backbone(imgs)         # (B,3488)

            # 2) ArcFace 로그릿
            logits = arc_head(emb, labels)  # (B,num_classes)

            # 3) CE Loss
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"[ArcFace][Epoch {epoch}] Avg Loss = {avg_loss:.4f}")

        # backbone만 따로 저장 (나중에 특징 추출용)
        torch.save(backbone.state_dict(), "checkpoints/arcface_backbone.pth")
        print("[INFO] Saved backbone to checkpoints/arcface_backbone.pth")

    print("[DONE] ArcFace training complete.")


if __name__ == "__main__":
    train_arcface()
