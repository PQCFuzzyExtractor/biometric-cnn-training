# dataset.py (최종 안정본)
import os
from PIL import Image
from torch.utils.data import Dataset


class CASIAIrisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "images")
        self.ids_csv = os.path.join(root_dir, "ids.csv")

        self.samples = []

        with open(self.ids_csv, "r", encoding="utf-8") as f:
            first_line = True

            for line in f:
                line = line.strip()

                # 빈 줄 skip
                if not line:
                    continue

                parts = line.split(",")

                # --------- 헤더 감지 및 skip ---------
                # 헤더는: ["", "Label", "ImagePath"]
                if first_line:
                    first_line = False
                    if "Label" in parts or "ImagePath" in parts:
                        # 헤더 줄은 건너뛰기
                        continue

                # --------- CSV 포맷 검사 ---------
                if len(parts) < 3:
                    print("⚠ CSV 형식 이상 감지(건너뜀):", parts)
                    continue

                # parts = [index, label, path]
                _, label_raw, full_path = parts

                # --------- PID 추출 (예: "437-R" → 437) ---------
                if "-" in label_raw:
                    pid_str = label_raw.split("-")[0]
                else:
                    pid_str = label_raw
                pid = int(pid_str)

                # --------- 경로 정리 ---------
                full_path = full_path.replace("\\", "/")

                key = "CASIA-Iris-Thousand/"

                if key not in full_path:
                    raise ValueError(f"[ERROR] 이미지 경로에서 기준 키워드를 찾지 못함:\n{full_path}\n키워드: {key}")

                # "CASIA-Iris-Thousand/" 뒤의 상대경로만 가져오기
                rel_path = full_path.split(key)[1]  # 예: "437/R/S5437R06.jpg"

                # 최종 이미지 파일 실제 위치
                img_path = os.path.join(self.images_dir, rel_path)

                self.samples.append((img_path, pid))

        # --------- PID → 이미지 리스트 매핑 ---------
        self.pid_to_imgs = {}
        for img, pid in self.samples:
            self.pid_to_imgs.setdefault(pid, []).append(img)

        print(f"[CASIAIrisDataset] Loaded {len(self.samples)} images from {len(self.pid_to_imgs)} subjects.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid = self.samples[idx]

        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, pid
