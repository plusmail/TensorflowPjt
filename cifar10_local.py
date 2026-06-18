import json

import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#학습 데이터(이미지) 경로 지정
train_dir = './cifar-10/train'
test_dir = './cifar-10/test'

#정답데이터 train만
train_cvs = './cifar-10/trainLabels.csv'
model_save_path = './saved_model/cifar10_model.keras'

# 결과물 사진
saved_path = './saved_model'
history_plot_path = saved_path + '/training_history.png'

class_map_save_path = saved_path + '/class_to_idx.json'

# 저장 경로 없으면 만들어 주기
os.makedirs(saved_path, exist_ok=True)

#정답 라벨링 읽어 오기 pandas
labels_df = pd.read_csv(train_cvs)

print(labels_df.head())
print(f'전체 학습 이미수 : {len(labels_df)}')

#unique는 중복 않함.
class_names = sorted(labels_df['label'].unique())

class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}
# {"airplane": 0,    }
print(class_to_idx)
print(idx_to_class)

with open(class_map_save_path, 'w', encoding='utf-8') as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
print(f'클래스 매핑 완료 : {class_map_save_path}')

# 학습용 데이터를 디렉토리에서 로딩
def load_images(image_dir , ids):
    images = []
    for img_id in ids:
        img_path = os.path.join(image_dir, f'{img_id}.png')
        img = Image.open(img_path).convert('RGB')
        images.append(np.array(img))

    return np.array(images)

x_train_full = load_images(train_dir, labels_df['id'].values)
y_train_full = labels_df['label'].map(class_to_idx).values

print(x_train_full.shape)
print(y_train_full.shape)

x_train_full = x_train_full.astype('float32') / 255.0  #정규화 0 ~ 1


#train에 있는 학습 자료 훈련용, 검증용으로 분할
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size = 0.2,
    random_state = 42,
    stratify= y_train_full
)

print(f'학습 : {x_train.shape}, 검증 : {x_val.shape}')

y_train_oh = to_categorical(y_train, 10)
y_val_oh = to_categorical(y_val, 10)


#가장 중요한 모델 설정
model = models.Sequential([
    layers.Conv2D(32, (3,3),
                  activation='relu',
                  padding='same',
                  input_shape=(32,32,3)
                  ),
    layers.MaxPool2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu',
                  padding='same'
                  ),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu',
                  padding='same'
                  ),
    layers.Flatten(), #1차원으로 변환
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

# 정답이 10개인 카테고리칼
    layers.Dense(10, activation='softmax')

])

#경사하강법 지정
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
#요약
model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#훈련(학습) 중 가장 좋은 부분을 저장
checkpoint_cb = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience= 5,
    restore_best_weights=True,
    verbose=1
)


history = model.fit(
    x_train, y_train_oh,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val_oh),
    callbacks=[checkpoint_cb, earlystop_cb]
)


model.save('./saved_model/cifar10_model_final.keras')
print(f'최종 모델 저장 완료 : ./saved_model 저장됨')


fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].plot(history.history['loss'], label='train_loss')
axes[0].plot(history.history['val_loss'], label='val_loss')
axes[0].set_title('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='train_acc')
axes[1].plot(history.history['val_accuracy'], label='val_acc')
axes[1].set_title('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig(history_plot_path)
plt.show()

val_loss, val_acc = model.evaluate(x_val, y_val_oh, verbose=0)
print(f"검증 Loss: {val_loss: .4f}, 검증 Accuracy : {val_acc:.4f} ")


