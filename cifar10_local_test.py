import json

import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

test_dir = './cifar-10/test'
model_load_path = './saved_model/cifar10_model.keras'
class_map_load_path ='./saved_model/class_to_idx.json'

submission_save_path = './my_submission.csv'
sample_pred_plot_path = './saved_model/test_predictions.png'

batch_size =1000

model = load_model(model_load_path)
print(f'모델 로딩 완료 :  {model_load_path}')

with open(class_map_load_path, 'r', encoding='utf-8') as f:
    class_to_idx = json.load(f)
idx_to_class = {int(idx) : name for name, idx in class_to_idx.items()}
print(f"클래스 매핑 로딩 완료 {idx_to_class}")

def load_images(image_dir , ids):
    images = []
    for img_id in ids:
        img_path = os.path.join(image_dir, f'{img_id}.png')
        img = Image.open(img_path).convert('RGB')
        images.append(np.array(img))

    return np.array(images)


test_files = sorted(
    [ f for f in os.listdir(test_dir) if f.endswith('.png')],
    key = lambda  x :  int(x.split('.')[0])
)

test_ids = [int(f.split('.')[0]) for f in test_files]

print(f'전체 테스트 이미지 수 : {len(test_ids)}')

def predict_in_batches(model, image_dir, ids, batch_size=1000):
    all_preds =[]
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        x_batch = load_images(image_dir, batch_ids)
        x_batch = x_batch.astype('float32') / 255.0
        preds = model.predict(x_batch, verbose=0)
        all_preds.append(preds)
        print(f'{start + len(batch_ids)} / {len(ids)} 완료')

    return np.concatenate(all_preds)


test_preds = predict_in_batches(model, test_dir, test_ids, batch_size=batch_size)
test_pred_labels = np.argmax(test_preds, axis=1)
test_pred_names = [idx_to_class[idx] for idx in test_pred_labels]

submission_df = pd.DataFrame({
    'id' : test_ids,
    'label': test_pred_labels
})

submission_df.to_csv(submission_save_path, index= False)
print(f'{submission_save_path} 저장 완료')
print(submission_df.head())

fig, axes = plt.subplots(2,5, figsize=(12,5))
sample_x = load_images(test_dir, test_ids[:10])

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_x[i])
    ax.set_title(f'pred: {test_pred_names[i]}')
    ax.axis('off')


plt.tight_layout()
plt.savefig(sample_pred_plot_path)
plt.show()

print(f'\n샘플 예측 결과 이미지 저장 완료 : {sample_pred_plot_path}')
print(f'테스트 완료')