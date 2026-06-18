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

MIN_CONTOUR_AREA = 1500  # 이 수자보다 작으면 무시
SMOOTHING = 0.3  # 0=즉시반은 1=거의 안움직임


def load_model_and_classes():
    model = load_model(model_load_path)
    with open(class_map_load_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
    return model, idx_to_class


def get_largest_motion_box(fg_mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


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

    # 배경 차분기
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=40, detectShadows=True
    )

    h_frame, w_frame = None, None
    smooth_box = None

    print("실시간 추론 시작. 종료하려면 'q'키를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        if h_frame is None:
            h_frame, w_frame = frame.shape[:2]

        # 배경처리 차분
        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 노이즈 제거 (작은 점들 ,,,
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((9, 9), np.uint8), iterations=2)

        motion_box = get_largest_motion_box(fg_mask)

        if motion_box is not None:
            x, y, w, h = motion_box

            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(w_frame - x, w + pad * 2)
            h = min(h_frame - y, h + pad * 2)

            if smooth_box is None:
                smooth_box = (x, y, w, h)
            else:
                sx, sy, sw, sh = smooth_box
                x = int(SMOOTHING * sx + (1 - SMOOTHING) * x)
                y = int(SMOOTHING * sy + (1 - SMOOTHING) * y)
                w = int(SMOOTHING * sw + (1 - SMOOTHING) * w)
                h = int(SMOOTHING * sh + (1 - SMOOTHING) * h)
                smooth_box = (x, y, w, h)

        if smooth_box is None:
            cv2.putText(frame, 'Waiting for....', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Motion Mask', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=40, detectShadows=True
                )
                print("배경 모델을 초기화 합니다.")
            continue

        x, y, w, h = smooth_box
        x0, y0 = x, y
        roi = frame[y0:y0 + h, x0:x0 + w]

        if roi.size > 0 and h > 10 and w > 10:
            x_input = preprocess_frame(roi)
            preds = model.predict(x_input, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_name = idx_to_class[top_idx]
            top_prob = float(preds[top_idx])
            cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)

            if top_prob >= CONF_THRESHOLD:
                label_text = f'{top_name}({top_prob * 100:.1f}%)'
                label_y = y0 - 10 if y0 - 10 > 10 else y0 + h + 25
                cv2.putText(frame, label_text, (x0, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            top3_idx = preds.argsort()[::-1][:3]
            for rank, idx in enumerate(top3_idx):
                text = f'{idx_to_class[idx]}: {preds[idx] * 100:.1f}%'
                cv2.putText(frame, text, (10, 30 + rank * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                            )
        else:
            smooth_box = None
            cv2.putText(frame, 'Waiting for....', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                        )

        cv2.imshow('Motion Mask', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=40, detectShadows=True
            )
            smooth_box = None
            print("배경 모델을 초기화 합니다.")

    cap.release()
    cv2.destroyAllWindows()
    print("종료합니다.")
    pass


if __name__ == '__main__':
    main()
