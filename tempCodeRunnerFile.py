# verify.py
import cv2
import sys
import numpy as np

import config
from db.face_db import FaceDatabase


def _safe_text(text: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")


def _db_path_display() -> str:
    try:
        return str(config.DB_PATH.relative_to(config.PROJECT_ROOT))
    except Exception:
        return str(config.DB_PATH.name)


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _face_quality_ok(face) -> bool:
    det_score = float(getattr(face, "det_score", 0.0))
    if det_score < float(getattr(config, "MIN_DET_SCORE", 0.0)):
        return False

    try:
        x1, y1, x2, y2 = face.bbox.astype(int)
        size = min(int(x2 - x1), int(y2 - y1))
    except Exception:
        return False

    return size >= int(getattr(config, "MIN_FACE_SIZE", 0))


def verify_from_camera():
    from collections import deque

    from face_detect import FaceDetector
    from liveness import LivenessDetector

    db = FaceDatabase(config.DB_PATH)
    total_embs = db.count_embeddings()
    templates = db.load_templates()
    usernames = sorted(str(u) for u in templates.keys())
    print(f"[INFO] DB Path: {_safe_text(_db_path_display())}")
    print(f"[INFO] Loaded {len(templates)} users ({total_embs} embeddings) from DB")
    if templates:
        if len(usernames) <= 20:
            users_line = ", ".join(usernames)
        else:
            users_line = ", ".join(usernames[:20]) + f", ... (+{len(usernames) - 20} more)"
        print(f"[INFO] Users in DB: {_safe_text(users_line)}")
    else:
        print("[WARN] No embeddings found. Run enroll.py first.")

    names = sorted(templates.keys())
    if names:
        mat = np.stack([templates[n] for n in names], axis=0)  # (N, D), normalized
    else:
        mat = None

    liveness = LivenessDetector() if config.LIVENESS_ENABLED else None
    liveness_model = None
    liveness_model_meta = None

    try:
        import torch
        from pathlib import Path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = getattr(config, "LIVENESS_MODEL_PATH", None)
        if model_path is not None:
            model_path = str(model_path)
            if model_path and Path(model_path).exists():
                from liveness_net import load_spoof_model

                liveness_model, liveness_model_meta = load_spoof_model(model_path, device)
                print(f"[INFO] Liveness model loaded: {_safe_text(model_path)}")
    except Exception:
        liveness_model = None
        liveness_model_meta = None

    spoof_fail_streak = 0
    live_pass_streak = 0
    spoof_state = False

    stable_name = "Unknown"
    stable_score = 0.0
    candidate_name = None
    candidate_score = 0.0
    candidate_count = 0

    try:
        detector_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        detector_device = "cpu"

    detector = FaceDetector(device=detector_device)

    cap = cv2.VideoCapture(0)
    emb_hist = deque(maxlen=int(getattr(config, "EMB_SMOOTH_FRAMES", 1)))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector.detect(frame)
        face = _largest_face(faces)
        if face is None:
            continue
        if not _face_quality_ok(face):
            continue

        # Face ROI for liveness: use raw crop (better texture than warped aligned face).
        face_roi = None
        try:
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2]
        except Exception:
            face_roi = None

        if liveness is not None and face_roi is not None:
            if liveness_model is not None and liveness_model_meta is not None:
                # Model-based anti-spoof (preferred when available).
                try:
                    import torch

                    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    s = int(liveness_model_meta["input_size"])
                    rgb = cv2.resize(rgb, (s, s))
                    x = rgb.astype(np.float32) / 255.0

                    mean = np.array(liveness_model_meta["mean"], dtype=np.float32)
                    std = np.array(liveness_model_meta["std"], dtype=np.float32)
                    x = (x - mean) / std
                    x = np.transpose(x, (2, 0, 1))

                    xt = torch.from_numpy(x).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logit = liveness_model(xt).squeeze(0).squeeze(0)
                        spoof_prob = float(torch.sigmoid(logit).item())

                    thresh = float(getattr(config, "LIVENESS_MODEL_SPOOF_THRESHOLD", 0.5))
                    live = spoof_prob < thresh
                    spoof_signal = not live

                    cv2.putText(frame, f"SpoofP {spoof_prob:.2f}", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception:
                    live = True
                    spoof_signal = False
            else:
                # Heuristic liveness (fallback).
                live, info = liveness.is_live(
                    face_roi,
                    lbp_thresh=config.LIVENESS_LBP_THRESH,
                    fft_thresh=config.LIVENESS_FFT_THRESH,
                )
                # Only treat as spoof when BOTH signals fail (reduces false positives).
                lbp_val = float(info.get("lbp", 0.0))
                fft_val = float(info.get("fft", 0.0))
                spoof_signal = (lbp_val < config.LIVENESS_LBP_THRESH) and (fft_val < config.LIVENESS_FFT_THRESH)

                cv2.putText(frame, f"LBP {lbp_val:.3f}  FFT {fft_val:.2f}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if live:
                live_pass_streak += 1
                spoof_fail_streak = 0
            else:
                live_pass_streak = 0
                if spoof_signal:
                    spoof_fail_streak += 1
                else:
                    spoof_fail_streak = max(0, spoof_fail_streak - 1)

            if not spoof_state and spoof_fail_streak >= config.LIVENESS_FAIL_FRAMES:
                spoof_state = True
                live_pass_streak = 0

            if spoof_state and live_pass_streak >= config.LIVENESS_PASS_FRAMES:
                spoof_state = False
                spoof_fail_streak = 0

            if spoof_state:
                cv2.putText(frame, "SPOOF DETECTED", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Verify", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            elif not live:
                cv2.putText(frame, "LIVENESS CHECK", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        emb = getattr(face, "embedding", None)
        if emb is None:
            continue

        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        emb = emb / norm
        emb_hist.append(emb)
        emb_use = np.mean(np.stack(list(emb_hist), axis=0), axis=0)
        n_use = np.linalg.norm(emb_use)
        if n_use != 0:
            emb_use = emb_use / n_use

        best_name = "Unknown"
        best_score = 0.0
        second_score = -1.0
        if mat is not None:
            scores = mat @ emb_use  # cosine, since both normalized
            if scores.size == 1:
                best_idx = 0
                best_score = float(scores[0])
                best_name = names[0]
            elif scores.size >= 2:
                top2 = np.argpartition(scores, -2)[-2:]
                i1, i2 = int(top2[0]), int(top2[1])
                if float(scores[i1]) >= float(scores[i2]):
                    best_idx, second_idx = i1, i2
                else:
                    best_idx, second_idx = i2, i1
                best_score = float(scores[best_idx])
                second_score = float(scores[second_idx])
                best_name = names[best_idx]

            if best_score < float(config.SIMILARITY_THRESHOLD):
                best_name = "Unknown"
            elif second_score >= 0 and (best_score - second_score) < float(getattr(config, "SCORE_GAP_THRESHOLD", 0.0)):
                best_name = "Unknown"

        if best_name == candidate_name:
            candidate_count += 1
            candidate_score = best_score
        else:
            candidate_name = best_name
            candidate_score = best_score
            candidate_count = 1

        if candidate_count >= config.NAME_STABLE_FRAMES:
            stable_name = candidate_name
            stable_score = candidate_score

        cv2.putText(frame, stable_name, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{stable_score:.2f}", (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Verify", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    db.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    verify_from_camera()
