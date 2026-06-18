"""
CIFAR-10 새 이미지 예측 스크립트
- 학습된 모델(cifar10_model.keras)을 로딩
- 사용자가 지정한 이미지 파일(들)을 예측
- CIFAR-10은 32x32 크기로 학습되었으므로, 어떤 크기의 이미지든 32x32로 리사이즈해서 추론

사용법:
    python predict_image.py 내사진.jpg
    python predict_image.py 사진1.jpg 사진2.png 사진3.jpg
"""

import sys
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =====================================================
# 경로 설정
# =====================================================
model_load_path = './saved_model/cifar10_model.keras'
class_map_load_path = './saved_model/class_to_idx.json'

IMG_SIZE = (32, 32)


# =====================================================
# 모델 및 클래스 매핑 로딩
# =====================================================
def load_model_and_classes():
    model = load_model(model_load_path)
    with open(class_map_load_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
    return model, idx_to_class


# =====================================================
# 이미지 전처리: 어떤 크기/포맷이든 32x32 RGB로 변환
# =====================================================
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)              # CIFAR-10 학습 크기로 리사이즈
    img_array = np.array(img).astype('float32') / 255.0
    return img_array


# =====================================================
# 예측 + 상위 N개 확률 출력
# =====================================================
def predict_image(model, idx_to_class, img_path, top_k=3):
    img_array = preprocess_image(img_path)
    x = np.expand_dims(img_array, axis=0)    # (1, 32, 32, 3) 배치 차원 추가

    preds = model.predict(x, verbose=0)[0]   # (10,) 클래스별 확률
    top_indices = preds.argsort()[::-1][:top_k]

    results = [(idx_to_class[idx], float(preds[idx])) for idx in top_indices]
    return img_array, results


# =====================================================
# 메인 실행
# =====================================================
def main():
    if len(sys.argv) < 2:
        print('사용법: python predict_image.py 이미지경로1 [이미지경로2 ...]')
        sys.exit(1)

    image_paths = sys.argv[1:]

    model, idx_to_class = load_model_and_classes()
    print(f'모델 로딩 완료. 클래스: {list(idx_to_class.values())}')

    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]   # subplot이 1개일 때도 리스트로 통일

    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f'파일 없음: {img_path}')
            continue

        img_array, results = predict_image(model, idx_to_class, img_path, top_k=3)

        print(f'\n[{img_path}]')
        for rank, (name, prob) in enumerate(results, 1):
            print(f'  {rank}위: {name} ({prob*100:.1f}%)')

        # 시각화
        axes[i].imshow(img_array)
        title_text = '\n'.join([f'{name}: {prob*100:.1f}%' for name, prob in results])
        axes[i].set_title(title_text, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    print('\nprediction_result.png 저장 완료')


if __name__ == '__main__':
    main()
