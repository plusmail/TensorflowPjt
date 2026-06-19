"""
predict_video.py (최적화 버전)
============================================================
듀얼 모델 실시간 추론 - 성능 개선판.

핵심 최적화
-----------
1) tf.function 그래프 컴파일       : eager 호출보다 빠름
2) N 프레임마다 추론 (frame skip)   : 화면은 부드럽게, 추론은 가끔만
3) 결과 변할 때만 "스티커" 재렌더   : 한글 PIL 변환을 매 프레임 하지 않음
4) 디스플레이 텍스트는 cv2.putText  : FPS 같은 영문/숫자는 OpenCV 네이티브

런타임 조작 키
-------------
  q / ESC  : 종료
  space    : 일시정지/재개
  s        : 스냅샷 저장
  + / -    : 추론 간격(N) 실시간 조절   (커질수록 빠르지만 반응 느림)
"""

import os
import json
import time
import argparse
import datetime

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 설정
# ============================================================
DUAL_MODEL_PATH = "my_mobilenet_dual.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
CUSTOM_CONFIDENCE_THRESHOLD = 0.6
TOP_K = 3

# 추론을 매 N 프레임마다 수행 (1=매프레임, 2=한 프레임 건너뜀, ...)
INFERENCE_EVERY_N_FRAMES = 3


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
    print("[경고] 한글 폰트 없음. 영문만 정상 표시.")
    return ImageFont.load_default()


# ============================================================
# tf.function 컴파일된 추론 (eager보다 빠름)
# ============================================================
@tf.function(reduce_retracing=True)
def _infer_compiled(model, x):
    return model(x, training=False)


def infer_one(dual_model, x):
    outputs = _infer_compiled(dual_model, x)
    return outputs[0].numpy(), outputs[1].numpy()


# ============================================================
# 전처리
# ============================================================
def preprocess_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    arr = np.expand_dims(resized.astype(np.float32), axis=0)
    return preprocess_input(arr)


def decode_custom(custom_pred, class_names):
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
# 결과 변할 때만 한 번 만드는 "오버레이 스티커"
# ============================================================
def render_overlay_sticker(imagenet_top, custom_label, custom_conf,
                           threshold, font_lg, font_md,
                           box_w=520):
    """
    검은 배경 위에 한글 텍스트를 그린 작은 BGR 이미지를 반환.
    매 프레임 PIL을 호출하지 않고, 추론이 일어났을 때만 이걸 1번 만든다.
    """
    box_h = 40 + (TOP_K + 1) * 28
    # PIL 캔버스 (RGB 검정)
    sticker_pil = Image.new("RGB", (box_w, box_h), (0, 0, 0))
    draw = ImageDraw.Draw(sticker_pil)

    # 최종 판단
    if custom_conf >= threshold:
        verdict = f"[사용자] {custom_label}  {custom_conf*100:.1f}%"
        verdict_color = (80, 255, 120)        # RGB 연두
    else:
        top1 = imagenet_top[0]
        verdict = f"[ImageNet] {top1[1]}  {top1[2]*100:.1f}%"
        verdict_color = (255, 200, 80)        # RGB 주황

    draw.text((12, 8), verdict, font=font_lg, fill=verdict_color)
    for i, (_, name, score) in enumerate(imagenet_top):
        line = f"{i+1}. {name}  {score*100:5.1f}%"
        draw.text((12, 40 + i * 26), line, font=font_md, fill=(230, 230, 230))
    user_line = f"   user: {custom_label}  {custom_conf*100:5.1f}%"
    draw.text((12, 40 + TOP_K * 26), user_line, font=font_md,
              fill=(160, 220, 255))

    # PIL(RGB) -> OpenCV(BGR)
    sticker = cv2.cvtColor(np.array(sticker_pil), cv2.COLOR_RGB2BGR)
    return sticker


def paste_sticker(frame, sticker, pos=(10, 10), alpha=0.55):
    """
    스티커를 프레임 위에 반투명 합성. (cv2.addWeighted 사용)
    매 프레임 호출되지만 PIL이 없어서 매우 빠름.
    """
    y, x = pos
    h, w = sticker.shape[:2]
    fh, fw = frame.shape[:2]
    h = min(h, fh - y)
    w = min(w, fw - x)
    if h <= 0 or w <= 0:
        return frame

    roi = frame[y:y + h, x:x + w]
    blended = cv2.addWeighted(sticker[:h, :w], alpha, roi, 1.0 - alpha, 0)
    frame[y:y + h, x:x + w] = blended
    return frame


# ============================================================
# 인자
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="듀얼 모델 영상 추론 (최적화)")
    p.add_argument("-s", "--source", default="0",
                   help="카메라 번호 또는 영상 파일 경로")
    p.add_argument("-m", "--model", default=DUAL_MODEL_PATH)
    p.add_argument("-c", "--classes", default=CLASS_NAMES_PATH)
    p.add_argument("-t", "--threshold", type=float,
                   default=CUSTOM_CONFIDENCE_THRESHOLD)
    p.add_argument("--save", default=None, help="결과 mp4 저장 경로")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("-n", "--every-n", type=int, default=INFERENCE_EVERY_N_FRAMES,
                   help=f"추론 주기 N (default: {INFERENCE_EVERY_N_FRAMES})")
    return p.parse_args()


# ============================================================
# 메인
# ============================================================
def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"모델 파일 없음: {args.model}")
    if not os.path.exists(args.classes):
        raise FileNotFoundError(f"클래스 파일 없음: {args.classes}")

    # ---- 모델 ----
    print(f"[로드] {args.model}")
    dual_model = tf.keras.models.load_model(args.model, compile=False)

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"[클래스] {class_names}")

    # ---- 워밍업 (tf.function 그래프 컴파일 트리거) ----
    print("[워밍업] 그래프 컴파일 중...")
    t0 = time.time()
    _ = infer_one(dual_model,
                  np.zeros((1,) + IMG_SIZE + (3,), dtype=np.float32))
    _ = infer_one(dual_model,
                  np.zeros((1,) + IMG_SIZE + (3,), dtype=np.float32))
    print(f"[워밍업] 완료 ({time.time() - t0:.2f}s)")

    # ---- 입력 ----
    src = args.source
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        # MJPG 설정: USB 캠에서 디코딩이 더 빠른 경우가 많음
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(src)

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 큐가 쌓여 레이턴시 늘어나는 것 방지

    if not cap.isOpened():
        raise RuntimeError(f"입력 열기 실패: {src}")

    # ---- 저장 ----
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        print(f"[저장] {args.save} {w}x{h} @ {fps:.1f}fps")

    font_lg = get_korean_font(22)
    font_md = get_korean_font(18)

    print("[시작] q/ESC=종료  space=일시정지  s=스냅샷  + / - =N 조절")

    paused = False
    frame_idx = 0
    cached_sticker = None
    every_n = max(1, args.every_n)

    fps_t0 = time.time()
    fps_count = 0
    fps_val = 0.0
    last_infer_ms = 0.0
    last_frame_for_pause = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[종료] 프레임 끝")
                break
            last_frame_for_pause = frame
        else:
            if last_frame_for_pause is None:
                break
            frame = last_frame_for_pause.copy()

        # ---- 추론은 N 프레임마다 한 번만 ----
        if not paused and (frame_idx % every_n == 0):
            t_i0 = time.time()
            x = preprocess_frame(frame)
            imagenet_pred, custom_pred = infer_one(dual_model, x)
            last_infer_ms = (time.time() - t_i0) * 1000.0

            imagenet_top = decode_predictions(imagenet_pred, top=TOP_K)[0]
            custom_label, custom_conf = decode_custom(custom_pred, class_names)

            # 결과가 바뀐 이번 추론에만 스티커 재렌더
            cached_sticker = render_overlay_sticker(
                imagenet_top, custom_label, custom_conf,
                args.threshold, font_lg, font_md
            )

        # ---- 스티커 합성 (PIL 없이, 매우 빠름) ----
        if cached_sticker is not None:
            frame = paste_sticker(frame, cached_sticker,
                                  pos=(10, 10), alpha=0.55)

        # ---- FPS ----
        fps_count += 1
        if fps_count >= 15:
            now = time.time()
            fps_val = fps_count / max(now - fps_t0, 1e-6)
            fps_t0 = now
            fps_count = 0

        # ---- 우상단 상태 표시 (cv2 네이티브) ----
        h_img, w_img = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps_val:5.1f}",
                    (w_img - 220, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"every N: {every_n}",
                    (w_img - 220, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"infer: {last_infer_ms:5.1f} ms",
                    (w_img - 220, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if paused:
            cv2.putText(frame, "[PAUSED]",
                        (w_img // 2 - 70, h_img - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Dual MobileNet (optimized)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
                print(f"[{'PAUSE' if paused else 'RESUME'}]")
            elif key == ord("s"):
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap = f"snapshot_{ts}.jpg"
                cv2.imwrite(snap, frame)
                print(f"[스냅샷] {snap}")
            elif key in (ord("+"), ord("=")):
                every_n = min(every_n + 1, 30)
                print(f"[every_n] {every_n}")
            elif key in (ord("-"), ord("_")):
                every_n = max(every_n - 1, 1)
                print(f"[every_n] {every_n}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[완료]")


if __name__ == "__main__":
    main()