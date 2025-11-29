# train_contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from dataset import CASIAIrisDataset
from model import IrisNet3488
from PIL import Image
import random
import os

class ContrastiveLoss(nn.Module):
    """ y=1 → 가까워야 함, y=0 → 멀어야 함 """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss_same = label * dist.pow(2)
        loss_diff = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return (loss_same + loss_diff).mean()


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    dataset = CASIAIrisDataset("data/CASIA", transform=None)

    # Create ID → list of image paths dict
    id_to_imgs = {}
    for img, pid in dataset.samples:
        id_to_imgs.setdefault(pid, []).append(img)
    pids = list(id_to_imgs.keys())

    model = IrisNet3488().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 3

    def load_image(path):
        img = Image.open(path).convert("L")
        return transform(img)

    for epoch in range(EPOCHS):
        model.train()

        for step in range(200):
            pid1, pid2 = random.sample(pids, 2)

            # same-person pair
            if len(id_to_imgs[pid1]) >= 2:
                A, B = random.sample(id_to_imgs[pid1], 2)
                imgA = load_image(A).unsqueeze(0).to(device)
                imgB = load_image(B).unsqueeze(0).to(device)
                label_same = torch.tensor([1.0]).to(device)

            # diff-person pair
            C = random.choice(id_to_imgs[pid2])
            imgC = load_image(C).unsqueeze(0).to(device)
            label_diff = torch.tensor([0.0]).to(device)

            outA = model(imgA)
            outB = model(imgB)
            outC = model(imgC)

            loss_same = criterion(outA, outB, label_same)
            loss_diff = criterion(outA, outC, label_diff)
            loss = loss_same + loss_diff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"[Epoch {epoch}] Step {step} Loss={loss.item():.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_contrastive_e{epoch}.pth")


if __name__ == "__main__":
    train()
