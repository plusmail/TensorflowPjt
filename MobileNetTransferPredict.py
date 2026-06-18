"""
학습된 MobileNetV2 전이학습 모델로 새 이미지를 예측하는 스크립트
================================================================

train_mobilenet_transfer.py 실행 후 생성된
my_mobilenet_model.h5, class_names.json 파일이 필요합니다.

실행 방법:
    python predict.py 테스트할이미지.jpg
"""

import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "my_mobilenet_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()

    # 이미지 로드 및 전처리 (학습 때와 동일하게)
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    predictions = model.predict(img_array, verbose=0)

    if predictions.shape[-1] == 1:
        # 이진 분류 (sigmoid)
        prob = float(predictions[0][0])
        idx = 1 if prob >= 0.5 else 0
        confidence = prob if idx == 1 else 1 - prob
        # 이진 분류일 땐 class_names[0]=음성클래스, class_names[1]=양성클래스 순서 확인 필요
        predicted_class = class_names[idx]
    else:
        # 다중 분류 (softmax)
        idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][idx])
        predicted_class = class_names[idx]

    print(f"\n이미지: {image_path}")
    print(f"예측 클래스: {predicted_class}")
    print(f"신뢰도: {confidence * 100:.2f}%")

    print("\n전체 클래스별 확률:")
    if predictions.shape[-1] == 1:
        prob = float(predictions[0][0])
        print(f"  {class_names[0]}: {(1 - prob) * 100:.2f}%")
        print(f"  {class_names[1]}: {prob * 100:.2f}%")
    else:
        for name, p in zip(class_names, predictions[0]):
            print(f"  {name}: {p * 100:.2f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict.py 이미지경로.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])