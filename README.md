# biometric-cnn-training
Iris Embedding with CNN + ArcFace (CASIA-Iris-Thousand)  
홍채 이미지를 입력으로 받아 **3488차원의 이진 특징 벡터**를 생성하고,  
벡터 간 **해밍 거리(Hamming Distance)** 를 이용해 같은 사람 / 다른 사람을 구분하는 프로젝트입니다.

최종적으로는 **ArcFace 기반 CNN 백본**이 가장 좋은 분리도를 보여  
Triplet Loss 기반 모델 대신 **ArcFace 모델을 최종 채택**했습니다.

---

## 📁 Dataset (CASIA-Iris-Thousand)

본 프로젝트에서는 공개 데이터셋 **CASIA-Iris-Thousand**를 사용합니다.

로컬 디렉토리 구조 예시는 다음과 같습니다:

```text
data/
 └── CASIA/
      ├── images/
      │    ├── 000/
      │    │    ├── L/*.jpg
      │    │    └── R/*.jpg
      │    ├── 001/
      │    └── ...
      └── ids.csv   # 각 이미지의 상대 경로, 사람 ID를 매핑
```

images : 원본 홍채 이미지 (좌/우, 피험자별 디렉토리)

ids.csv : relative_path,person_id 형식으로 이미지와 피험자 ID 매핑


## 🧠 Model Overview
### 출력 벡터  
CNN 백본 + ArcFace 헤드를 통해

3488차원의 실수 특징 벡터를 출력

이후 sign() 또는 임계값(0 기준)으로 이진화하여 3488비트 벡터로 사용

최종적으로 이 3488비트 벡터는 해밍 거리 기반 비교, 퍼지 추출기(Fuzzy Extractor), McEliece(3488, 64) 같은 후단 모듈에서 사용 가능하도록 설계

### 실험한 학습 방식
#### Triplet Loss 기반 임베딩 모델

Anchor / Positive / Negative 샘플 조합으로 학습

상대적인 거리(Anchor–Positive vs Anchor–Negative)를 기준으로 학습

기본적인 분리 성능은 양호

#### ArcFace 기반 모델 (최종 선택)

Angular Margin 기반 분류(ArcFace) + CNN 백본

같은 클래스(같은 사람) 임베딩은 적당히 모으고,

다른 클래스(다른 사람) 간 거리를 크게 벌려주는 특성

Triplet보다 서로 다른 사람 간의 거리(분리도)가 더 크게 확보되어 최종 채택

## 🚀 Training
ArcFace 기반 학습은 다음 명령어로 수행합니다:

```bash
python train_arcface.py
```
학습이 완료되면 체크포인트가 저장됩니다.

```text
checkpoints/arcface_backbone.pth
```
이 파일이 홍채 임베딩을 생성하는 최종 CNN 백본 모델입니다.

입력 이미지 → 3488차원 특징 벡터까지를 책임지는 부분입니다.

## 🧪 Evaluation (Hamming Distance)
학습된 ArcFace 백본을 이용해
같은 사람 / 다른 사람 쌍의 해밍 거리 통계를 계산하는 스크립트입니다.

```bash
python evaluate_arcface.py
```
예시 출력:

```text
[INFO] Using device: cuda
[CASIAIrisDataset] Loaded 20000 images from 1000 subjects.
[INFO] Loaded ArcFace backbone from checkpoints/arcface_backbone.pth
[INFO] Total samples: 20000
---- ArcFace Evaluation Result ----
Same person HD avg : 552.21
Diff person HD avg : 1728.19
(num_pairs=2000)
Same person HD avg : 같은 사람(동일 피험자) 이미지 쌍의 평균 해밍 거리

Diff person HD avg : 서로 다른 사람 이미지 쌍의 평균 해밍 거리
```

거리 차이가 클수록(분리도가 높을수록) 실제 인증 단계에서 유리합니다.

## 🎯 Performance Summary (Triplet vs ArcFace)
동일한 CASIA-Iris-Thousand 데이터셋을 사용하여
Triplet Loss 기반 모델과 ArcFace 기반 모델을 비교했습니다.

| Model | Same-person HD avg ↓ | Diff-person HD avg ↑ | Separation (Diff - Same) ↑ |
| :--- | :---: | :---: | :---: |
| Triplet (baseline) | ~522 | ~1468 | ~945 |
| ArcFace (final) | ~552 | ~1728 | ~1176 | 

Triplet은 기본적으로 양호한 성능을 보였으나,

ArcFace는 서로 다른 사람 간 거리를 훨씬 멀게(≈ 1728) 만들면서
분리도(Separation)가 더 크기 때문에 최종 모델로 선택했습니다.

## 🔐 “Hamming Distance”에 대한 설명
### 전통적인 홍채 인식(IrisCode)에서는:

홍채 이미지를 고정 길이의 이진 코드(예: 2048bit)로 변환한 뒤,

두 코드 간의 서로 다른 비트 개수를 세어 Hamming Distance를 계산합니다.

이때 값은 보통 0.0 ~ 0.5 범위의 정규화된 수치로 표현됩니다.

### 본 프로젝트에서는:

CNN + ArcFace 백본이 생성한 3488차원 벡터를 이진화한 뒤,

두 벡터의 비트가 다른 개수를 세어 “해밍 거리”와 같은 방식으로 사용합니다.

벡터 길이가 3488이기 때문에, 이론상 해밍 거리 범위는 0 ~ 3488입니다.

Same-person가 수백 대, Diff-person가 천 단위인 값이 나오는 것은 자연스러운 결과입니다.

즉, “전통적인 IrisCode를 그대로 구현한 것은 아니지만,
3488비트 벡터 간의 차이를 해밍 거리와 동일한 방식으로 계산해 사용한다.”

라는 의미입니다.

## 📦 Repository Structure
예시:

```text
.
├── data/
│   └── CASIA/
│       ├── images/
│       └── ids.csv
├── checkpoints/
│   ├── arcface_backbone.pth
│   └── (optional) triplet_model.pth
├── dataset.py          # CASIA iris dataset loader
├── model_arcface.py    # ArcFace 기반 CNN 백본 및 헤드
├── losses.py           # ArcFace / Triplet 등 loss 구현
├── train_arcface.py    # ArcFace 학습 스크립트
├── train_triplet.py    # Triplet 학습 스크립트 (baseline)
├── evaluate_arcface.py # ArcFace 백본을 이용한 HD 평가
├── evaluate.py         # Triplet 모델용 평가 스크립트 (optional)
└── utils.py            # 공통 유틸 함수
```

## 📜 License
Dataset: CASIA-Iris-Thousand (CASIA 측 라이선스 및 사용 규정 준수 필요)

Code: 연구 및 학습용(Research / Academic) 목적으로 사용
