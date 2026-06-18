import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

DATASET_DIR = "./dataset/train"
VAL_DIR = "./dataset/validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 5

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_HEAD = 1e-5
FINE_TUNE_AT_LAYER = 100  # 이 레이어 이후만 학습해라.
MODEL_SAVE_PATH = 'my_mobilenet_model.h5'
CLASS_NAMES_PATH = 'class_names.json'


def build_datasets():
    has_val_dir = os.path.isdir(VAL_DIR)
    if has_val_dir:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed = 42,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            subset='training',
            seed = 42,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            subset='validation',
            seed = 42,
        )

    class_names = train_ds.class_names
    print(f"클랙스 목록 :  {class_names}")

    with open(CLASS_NAMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name='data_augmentation')

    #내장함수
    def prepare(ds, training=False):
        if training:
            ds = ds.map(lambda x,y : (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE) # cpu 병렬활용

        ds = ds.map(lambda x, y : (preprocess_input(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                    )

        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = prepare(train_ds, training=True)
    val_ds = prepare(val_ds, training=False)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    """
    MobileNetV2 베이스 모델 + 사용자 정의 분류기(head)를 구성합니다.
    """
    # ImageNet으로 사전학습된 MobileNetV2를 불러옴
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),  # (224, 224, 3) -> 가로,세로,채널(RGB)
        include_top=False,  # 기존 1000개 클래스 분류기는 제거
        weights="imagenet",  # ImageNet 사전학습 가중치 사용
    )
    # 1단계에서는 베이스 모델의 모든 가중치를 학습 대상에서 제외 (역전파로 갱신되지 않음)
    base_model.trainable = False  # 1단계: 베이스 모델 가중치 고정

    # 모델의 입력 텐서 정의 (224,224,3) 크기의 이미지를 받음
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # 입력 이미지를 베이스 모델에 통과시켜 특징(feature map) 추출
    # training=False: 베이스 모델 내부의 BatchNorm 레이어가 학습 모드로 흔들리지 않도록 고정
    x = base_model(inputs, training=False)
    # 베이스 모델이 뽑아낸 (H, W, C) 형태의 특징맵을 채널별 평균으로 압축 -> (C,) 벡터로 변환
    x = layers.GlobalAveragePooling2D()(x)
    # 과적합 방지를 위해 노드의 30%를 무작위로 비활성화 (학습 시에만 적용됨)
    x = layers.Dropout(0.3)(x)
    # 새로 추가하는 완전연결층: 128개 노드, ReLU 활성화로 비선형성 부여
    x = layers.Dense(128, activation="relu")(x)
    # 두 번째 드롭아웃: 20%를 비활성화 (Dense 레이어 뒤에서 추가 정규화)
    x = layers.Dropout(0.2)(x)

    # 클래스가 2개(이진 분류)인지, 3개 이상(다중 분류)인지에 따라 출력층 구성을 다르게 함
    if num_classes == 2:
        # 출력 노드 1개 + sigmoid: 0~1 사이 확률값 하나로 "양성 클래스일 확률"을 표현
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"  # 이진 분류에 맞는 손실 함수
    else:
        # 출력 노드 num_classes개 + softmax: 모든 클래스에 대한 확률 분포(합=1)를 출력
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"  # 레이블이 정수(0,1,2,...)일 때 쓰는 손실 함수

    # 정의한 inputs -> outputs 연결 구조를 하나의 Model 객체로 묶음
    model = models.Model(inputs, outputs)
    # 완성된 모델, (미세조정 단계에서 다시 조작할) base_model, 사용한 손실함수명을 함께 반환
    return model, base_model, loss


def plot_history(history_head, history_fine, save_path="training_history.png"):
    """학습 곡선을 이미지로 저장합니다."""
    # 1단계와 2단계의 학습 정확도 기록을 하나의 리스트로 이어붙임 (그래프를 연속적으로 보기 위함)
    acc = history_head.history["accuracy"] + history_fine.history["accuracy"]
    # 검증 정확도도 동일하게 이어붙임
    val_acc = history_head.history["val_accuracy"] + history_fine.history["val_accuracy"]
    # 학습 손실 이어붙이기
    loss = history_head.history["loss"] + history_fine.history["loss"]
    # 검증 손실 이어붙이기
    val_loss = history_head.history["val_loss"] + history_fine.history["val_loss"]

    # 1단계가 몇 에폭에서 끝났는지를 기록 -> 그래프에 "여기서 미세조정 시작"이라는 구분선을 그릴 위치
    switch_epoch = len(history_head.history["accuracy"])

    # 가로 12, 세로 5 크기의 그래프 도화지 생성
    plt.figure(figsize=(12, 5))

    # 첫 번째 서브플롯(좌측): 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")  # 학습 정확도 곡선
    plt.plot(val_acc, label="Validation Accuracy")  # 검증 정확도 곡선
    # 1단계->2단계 전환 지점에 점선 표시
    plt.axvline(switch_epoch - 1, color="gray", linestyle="--", label="Fine-tune 시작")
    plt.legend()
    plt.title("Accuracy")

    # 두 번째 서브플롯(우측): 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(switch_epoch - 1, color="gray", linestyle="--", label="Fine-tune 시작")
    plt.legend()
    plt.title("Loss")

    # 두 서브플롯 간 여백을 자동으로 깔끔하게 정리
    plt.tight_layout()
    # 화면에 띄우는 대신 파일로 저장 (서버/터미널 환경에서도 동작하도록)
    plt.savefig(save_path)
    print(f"학습 곡선 저장됨: {save_path}")

def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f'{DATASET_DIR} 폴더가 없습니다.'
            f'dataset/train/클래스명/이미지.jpg 형태로 이미지를 넣어주세요.'
        )

    train_ds, val_ds, class_names = build_datasets()
    num_classes = len(class_names)

    model , base_model, loss_fn = build_model(num_classes)

    #1단계
    print("1단계 분류기 학습")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss= loss_fn,
        metrics=["accuracy"]
    )

    model.summary()

    early_top = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=[early_top]
    )

    #2단계  미세조정(Fine-tuning) 베이스 모델 일부 레이어 학습 허용
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_HEAD),
        loss = loss_fn,
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=[early_top]
    )

    model.save(MODEL_SAVE_PATH)
    print(f'\n모델 저장 완료 {MODEL_SAVE_PATH}')
    print(f'클래스 이름 저장 완료: {CLASS_NAMES_PATH}')
    # 1단계+2단계 학습 곡선을 하나의 이미지로 그려서 저장
    plot_history(history_head, history_fine)


    val_loss, val_acc = model.evaluate(val_ds)
    print(f'\n 최종 검증 정확도: {val_acc:.4f} / 검증 손실: {val_loss:.4f}')




    pass


if __name__ == "__main__":
    main()
