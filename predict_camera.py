import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model_path = './saved_model/'
model_load_path = model_path + 'cifar10_model_final.keras'
class_map_load_path = model_path + 'class_to_idx.json'

IMG_SIZE = (32, 32)
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.0


def load_model_and_classes():
    model = load_model(model_load_path)
    with open(class_map_load_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)

    idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}

    return model, idx_to_class


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)  # (1,32,32,3)


def main():
    model, idx_to_class = load_model_and_classes()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f'카메라를 열 수 없습니다.')
        return

    print("실시간 추론 시작. 종료하려면 'q'키를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        roi = frame[y0:y0 + side, x0:x0 + side]

        x = preprocess_frame(roi)

        preds = model.predict(x, verbose=0)[0]
        print(preds)

        top_idx = int(np.argmax(preds))
        top_name = idx_to_class[top_idx]
        top_prob = float(preds[top_idx])

        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)

        if top_prob >= CONF_THRESHOLD:
            label_text = f'{top_name}({top_prob * 100:.1f}%)'
            cv2.putText(frame, label_text, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        top3_idx = preds.argsort()[::-1][:3]
        for rank, idx in enumerate(top3_idx):
            text = f'{idx_to_class[idx]}: {preds[idx] * 100:.1f}%'
            cv2.putText(frame, text, (10, 30 + rank * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                        )

        cv2.imshow('CIFAR-10', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("종료합니다.")
    pass


if __name__ == '__main__':
    main()
