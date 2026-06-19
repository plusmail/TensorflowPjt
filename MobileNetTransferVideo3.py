"""
predict_video.py  -  Grad-CAM 객체 위치 박스 표시
============================================================
MobileNetV2 분류기는 박스를 직접 못 내놓습니다. 그래서:
  1) custom_model 의 마지막 conv feature map 을 노출하는 grad_model 빌드
  2) 예측 클래스 점수에 대한 conv 기울기 -> 채널 중요도 -> 가중합 = 히트맵
  3) 히트맵을 프레임 크기로 업샘플 -> 임계값 -> 최대 영역의 bounding box

조작 키
-------
  q / ESC : 종료
  space   : 일시정지/재개
  s       : 스냅샷
  b       : 박스 ON/OFF
  h       : 히트맵 ON/OFF
  + / -   : 추론 간격(N) 조절
"""

import os
import json
import time
import argparse
import datetime

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 설정
# ============================================================
CUSTOM_MODEL_PATH = "my_mobilenet_custom.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
CUSTOM_CONFIDENCE_THRESHOLD = 0.6
TOP_K = 3
INFERENCE_EVERY_N_FRAMES = 6

# Grad-CAM
GRADCAM_THRESHOLD = 0.5      # 히트맵 임계값 (0~1)
MIN_BBOX_AREA_RATIO = 0.02    # 프레임 대비 최소 박스 면적 비율

# 색상 (OpenCV는 BGR)
COLOR_CUSTOM = (120, 255, 80)    # 연두
COLOR_IMAGENET = (80, 200, 255)  # 주황


# ============================================================
# 한글 폰트
# ============================================================
def get_korean_font(size=20):
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "./NanumGothic.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    print("[경고] 한글 폰트 없음")
    return ImageFont.load_default()


# ============================================================
# Grad-CAM 모델 빌드
# ============================================================
def build_gradcam_model(custom_model):
    """
    custom_model 의 입력 ->
      1) 마지막 conv feature map (nested MobileNetV2 의 출력)
      2) 사용자 분류 예측
    둘 다 한 번에 내놓는 모델.

    ⚠ 핵심: predictions 가 conv_features 텐서를 통과해서 계산되어야
       tape.gradient(predictions, conv_features) 가 valid 한 값을 돌려준다.
       그래서 custom_model 을 통째로 한 번 더 호출하면 안 되고, head 레이어들을
       conv_features 위에 직접 다시 얹어서 '하나의 그래프'로 잇는다.
    """
    inputs = custom_model.input

    # custom_model 안의 nested MobileNetV2 찾기 (인덱스도 같이)
    base_model = None
    base_idx = None
    for i, layer in enumerate(custom_model.layers):
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            base_idx = i
            break
    if base_model is None:
        raise RuntimeError("nested MobileNetV2 base 를 찾을 수 없음")

    # 1) base 한 번 통과 -> conv feature map
    conv_features = base_model(inputs, training=False)   # (1, 7, 7, 1280)

    # 2) base 뒤에 오는 head 레이어들을 conv_features 위에 다시 얹기
    #    (GlobalAveragePooling2D -> Dropout -> Dense -> Dropout -> Dense)
    x = conv_features
    for layer in custom_model.layers[base_idx + 1:]:
        x = layer(x)
    predictions = x

    # 이제 predictions 는 conv_features 를 거쳐서 만들어진 텐서이므로
    # 기울기가 conv_features 까지 잘 흘러간다.
    return Model(
        inputs=inputs,
        outputs=[conv_features, predictions],
        name="gradcam_model",
    )


# ============================================================
# Grad-CAM 계산 (그래프 컴파일)
# ============================================================
@tf.function(reduce_retracing=True)
def _gradcam_compiled(grad_model, x):
    """
    한 번의 forward + 한 번의 backward 로
      cam(H, W), preds(1, K), class_idx() 를 반환.
    """
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        # conv_out: (1, H, W, C), preds: (1, K) 또는 (1, 1) for binary
        if preds.shape[-1] == 1:
            target = preds[0, 0]
            class_idx = tf.constant(0, dtype=tf.int32)
        else:
            class_idx = tf.cast(tf.argmax(preds[0]), tf.int32)
            target = preds[0, class_idx]

    grads = tape.gradient(target, conv_out)           # (1, H, W, C)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)  공간 평균 = 채널 중요도
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # (H, W)  가중합
    cam = tf.nn.relu(cam)                              # 음수 영향은 버림
    return cam, preds, class_idx


def gradcam_to_bboxes_and_heatmap(cam_np, frame_shape,
                                  threshold=GRADCAM_THRESHOLD,
                                  min_area_ratio=MIN_BBOX_AREA_RATIO,
                                  max_boxes=5,
                                  close_ksize=15):
    """
    cam (작은 H,W) -> 프레임 크기 히트맵 + bounding box 여러 개.

    동작 흐름:
      1) 히트맵 0~1 정규화 후 프레임 크기로 업샘플
      2) threshold 이상 영역만 마스크 (binary)
      3) 모폴로지 close 로 가까운 작은 점들을 합쳐 큰 덩어리로 만듦
      4) findContours -> 면적이 min_area_ratio 이상인 것만 채택
      5) 면적 큰 순으로 max_boxes 개까지 반환 (각 박스의 peak score 함께)
    """
    h_frame, w_frame = frame_shape[:2]
    cam_max = float(cam_np.max())
    if cam_max < 1e-6:
        return [], None

    cam_norm = (cam_np / cam_max).astype(np.float32)
    cam_resized = cv2.resize(cam_norm, (w_frame, h_frame))

    # 1) 마스크
    mask = (cam_resized >= threshold).astype(np.uint8) * 255

    # 2) 모폴로지 close: 가까운 hotspot 들을 한 덩어리로
    if close_ksize and close_ksize > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3) 외곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 4) 면적 필터 + 정렬 + 상위 N개
    min_area = h_frame * w_frame * min_area_ratio
    cand = [c for c in contours if cv2.contourArea(c) >= min_area]
    cand.sort(key=cv2.contourArea, reverse=True)
    cand = cand[:max_boxes]

    # 5) 각 박스에 대해 peak score 도 같이 계산
    bboxes = []
    for c in cand:
        x, y, w, h = cv2.boundingRect(c)
        roi = cam_resized[y:y + h, x:x + w]
        score = float(roi.max()) if roi.size > 0 else 0.0
        bboxes.append({"rect": (x, y, w, h), "score": score})

    # 컬러 히트맵 (옵션 표시용)
    cam_u8 = (cam_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    return bboxes, heatmap_colored


def draw_bbox(frame, bbox, color_bgr, thickness=3, label=None):
    """
    모서리 강조된 디텍션 스타일 박스.
    label 이 있으면 박스 위에 작은 ASCII 텍스트 라벨도 그림.
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, thickness)

    # 4 모서리 L자 강조
    L = max(15, min(w, h) // 6)
    t2 = thickness + 2
    for (cx, cy, dx, dy) in [
        (x,     y,     +1, +1),
        (x + w, y,     -1, +1),
        (x,     y + h, +1, -1),
        (x + w, y + h, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color_bgr, t2)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color_bgr, t2)

    # 라벨 (ASCII 만; 한글은 좌상단 스티커에서 보여줌)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly0 = max(y - th - 8, 0)
        cv2.rectangle(frame, (x, ly0), (x + tw + 10, y), color_bgr, -1)
        cv2.putText(frame, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


# ============================================================
# 출력 해석
# ============================================================
def decode_custom_from_preds(custom_pred, class_names):
    if custom_pred.shape[-1] == 1:
        p = float(custom_pred[0, 0])
        if p >= 0.5:
            return class_names[1], p
        else:
            return class_names[0], 1.0 - p
    else:
        idx = int(np.argmax(custom_pred[0]))
        return class_names[idx], float(custom_pred[0, idx])


# ============================================================
# 전처리
# ============================================================
def preprocess_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    arr = np.expand_dims(resized.astype(np.float32), axis=0)
    return preprocess_input(arr)


# ============================================================
# 한글 스티커 (top-left 텍스트)
# ============================================================
def render_overlay_sticker(imagenet_top, custom_label, custom_conf,
                           threshold, font_lg, font_md,
                           box_w=520):
    box_h = 40 + (TOP_K + 1) * 28
    sticker_pil = Image.new("RGB", (box_w, box_h), (0, 0, 0))
    draw = ImageDraw.Draw(sticker_pil)

    if custom_conf >= threshold:
        verdict = f"[사용자] {custom_label}  {custom_conf*100:.1f}%"
        vcolor = (80, 255, 120)
    else:
        top1 = imagenet_top[0]
        verdict = f"[ImageNet] {top1[1]}  {top1[2]*100:.1f}%"
        vcolor = (255, 200, 80)

    draw.text((12, 8), verdict, font=font_lg, fill=vcolor)
    for i, (_, name, score) in enumerate(imagenet_top):
        draw.text((12, 40 + i * 26),
                  f"{i+1}. {name}  {score*100:5.1f}%",
                  font=font_md, fill=(230, 230, 230))
    draw.text((12, 40 + TOP_K * 26),
              f"   user: {custom_label}  {custom_conf*100:5.1f}%",
              font=font_md, fill=(160, 220, 255))
    return cv2.cvtColor(np.array(sticker_pil), cv2.COLOR_RGB2BGR)


def paste_sticker(frame, sticker, pos=(10, 10), alpha=0.55):
    y, x = pos
    h, w = sticker.shape[:2]
    fh, fw = frame.shape[:2]
    h = min(h, fh - y); w = min(w, fw - x)
    if h <= 0 or w <= 0:
        return frame
    roi = frame[y:y + h, x:x + w]
    blended = cv2.addWeighted(sticker[:h, :w], alpha, roi, 1.0 - alpha, 0)
    frame[y:y + h, x:x + w] = blended
    return frame


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="듀얼 모델 + Grad-CAM 박스 추론")
    p.add_argument("-s", "--source", default="0",
                   help="카메라 번호 또는 영상 파일")
    p.add_argument("-m", "--model", default=CUSTOM_MODEL_PATH,
                   help=f"사용자 분류기 (default: {CUSTOM_MODEL_PATH})")
    p.add_argument("-c", "--classes", default=CLASS_NAMES_PATH)
    p.add_argument("-t", "--threshold", type=float,
                   default=CUSTOM_CONFIDENCE_THRESHOLD,
                   help="사용자 분류기 신뢰도 임계값")
    p.add_argument("--cam-threshold", type=float,
                   default=GRADCAM_THRESHOLD,
                   help=f"Grad-CAM 히트맵 임계값 (default: {GRADCAM_THRESHOLD})")
    p.add_argument("--max-boxes", type=int, default=3,
                   help="동시에 그릴 최대 박스 개수 (default: 3)")
    p.add_argument("--close-ksize", type=int, default=15,
                   help="모폴로지 close 커널 크기 px. 클수록 가까운 박스를 더 합침 (0=끄기)")
    p.add_argument("--save", default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("-n", "--every-n", type=int,
                   default=INFERENCE_EVERY_N_FRAMES)
    return p.parse_args()


# ============================================================
# 메인
# ============================================================
def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"사용자 모델 없음: {args.model}")
    if not os.path.exists(args.classes):
        raise FileNotFoundError(f"클래스 파일 없음: {args.classes}")

    # ---- 모델 로드 ----
    print(f"[로드] 사용자 분류기 {args.model}")
    custom_model = tf.keras.models.load_model(args.model, compile=False)

    print("[로드] ImageNet MobileNetV2")
    imagenet_model = MobileNetV2(weights="imagenet", include_top=True)
    imagenet_model.trainable = False

    print("[빌드] Grad-CAM 모델")
    grad_model = build_gradcam_model(custom_model)

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"[클래스] {class_names}")

    # ---- 워밍업 (그래프 컴파일) ----
    print("[워밍업] 그래프 컴파일 중...")
    t0 = time.time()
    dummy = tf.constant(np.zeros((1,) + IMG_SIZE + (3,), dtype=np.float32))
    _ = imagenet_model(dummy, training=False); _ = imagenet_model(dummy, training=False)
    _ = _gradcam_compiled(grad_model, dummy); _ = _gradcam_compiled(grad_model, dummy)
    print(f"[워밍업] {time.time() - t0:.2f}s")

    # ---- 입력 ----
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(src)
    if args.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"입력 열기 실패: {src}")

    # ---- 저장 ----
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        print(f"[저장] {args.save} {w}x{h}@{fps:.1f}")

    font_lg = get_korean_font(22)
    font_md = get_korean_font(18)

    print("[시작] q=종료 space=일시정지 s=스냅 b=박스 h=히트맵 +/-=N")

    # 상태
    paused = False
    show_box = True
    show_heatmap = False
    frame_idx = 0
    every_n = max(1, args.every_n)

    cached_sticker = None
    cached_bboxes = []           # ← 리스트로 변경
    cached_heatmap = None
    cached_color = COLOR_CUSTOM

    fps_t0 = time.time(); fps_count = 0; fps_val = 0.0
    last_infer_ms = 0.0
    last_frame_for_pause = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[종료] 프레임 끝"); break
            last_frame_for_pause = frame
        else:
            if last_frame_for_pause is None: break
            frame = last_frame_for_pause.copy()

        # ---- 추론 (N 프레임마다) ----
        if not paused and (frame_idx % every_n == 0):
            t_i0 = time.time()
            x = preprocess_frame(frame)
            x_t = tf.constant(x)

            # ImageNet 분류
            imagenet_pred = imagenet_model(x_t, training=False).numpy()

            # 사용자 분류 + Grad-CAM (한 번에)
            cam_t, preds_t, _ = _gradcam_compiled(grad_model, x_t)
            cam_np = cam_t.numpy()
            custom_pred = preds_t.numpy()

            last_infer_ms = (time.time() - t_i0) * 1000.0

            # 결과 해석
            imagenet_top = decode_predictions(imagenet_pred, top=TOP_K)[0]
            custom_label, custom_conf = decode_custom_from_preds(
                custom_pred, class_names)

            # 색상
            cached_color = (COLOR_CUSTOM if custom_conf >= args.threshold
                            else COLOR_IMAGENET)

            # bbox 들 + heatmap 캐시
            cached_bboxes, cached_heatmap = gradcam_to_bboxes_and_heatmap(
                cam_np, frame.shape,
                threshold=args.cam_threshold,
                max_boxes=args.max_boxes,
                close_ksize=args.close_ksize,
            )

            # 스티커 캐시
            cached_sticker = render_overlay_sticker(
                imagenet_top, custom_label, custom_conf,
                args.threshold, font_lg, font_md)

        # ---- 시각화 ----
        if show_heatmap and cached_heatmap is not None:
            frame = cv2.addWeighted(frame, 0.6, cached_heatmap, 0.4, 0)

        if show_box and cached_bboxes:
            for i, bb in enumerate(cached_bboxes):
                rect = bb["rect"]
                score = bb["score"]
                # 1등 박스는 더 두껍게, 그 외는 살짝 얇게
                thick = 3 if i == 0 else 2
                # 박스마다 점수 라벨 (#1 87%, #2 64% ...)
                label = f"#{i+1} {score*100:.0f}%"
                draw_bbox(frame, rect, cached_color, thickness=thick,
                          label=label)

        if cached_sticker is not None:
            frame = paste_sticker(frame, cached_sticker, (10, 10), 0.55)

        # FPS
        fps_count += 1
        if fps_count >= 15:
            now = time.time()
            fps_val = fps_count / max(now - fps_t0, 1e-6)
            fps_t0 = now; fps_count = 0

        h_img, w_img = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps_val:5.1f}",
                    (w_img - 230, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"every N: {every_n}",
                    (w_img - 230, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"infer: {last_infer_ms:5.1f} ms",
                    (w_img - 230, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"box:{'ON' if show_box else 'OFF'} "
                    f"heat:{'ON' if show_heatmap else 'OFF'}",
                    (w_img - 230, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)

        if paused:
            cv2.putText(frame, "[PAUSED]",
                        (w_img // 2 - 70, h_img - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Dual MobileNet + Grad-CAM", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27: break
            elif key == ord(" "): paused = not paused
            elif key == ord("s"):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"snapshot_{ts}.jpg", frame)
                print(f"[스냅샷] snapshot_{ts}.jpg")
            elif key == ord("b"):
                show_box = not show_box
                print(f"[box] {'ON' if show_box else 'OFF'}")
            elif key == ord("h"):
                show_heatmap = not show_heatmap
                print(f"[heatmap] {'ON' if show_heatmap else 'OFF'}")
            elif key in (ord("+"), ord("=")):
                every_n = min(every_n + 1, 30); print(f"[N] {every_n}")
            elif key in (ord("-"), ord("_")):
                every_n = max(every_n - 1, 1); print(f"[N] {every_n}")

        frame_idx += 1

    cap.release()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()
    print("[완료]")


if __name__ == "__main__":
    main()