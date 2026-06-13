# enroll.py
import argparse
import sys

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


def enroll_from_camera(name, replace_existing: bool = False):
    import cv2
    import numpy as np

    from face_detect import FaceDetector

    db = FaceDatabase(config.DB_PATH)
    print(f"[INFO] DB Path: {_safe_text(_db_path_display())}")
    if replace_existing:
        deleted = db.delete_user(name)
        print(f"[INFO] Replacing existing user. Deleted {deleted} old embeddings for: {_safe_text(name)}")

    try:
        import torch

        detector_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        detector_device = "cpu"

    detector = FaceDetector(device=detector_device)

    cap = cv2.VideoCapture(0)
    # Lower resolution to speed up CPU inference
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    candidates = []
    target = int(getattr(config, "ENROLL_SAMPLES", config.NUM_ENROLL_EMBEDDINGS))

    while len(candidates) < target:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector.detect(frame)
        face = _largest_face(faces)
        if face is None:
            continue
        if not _face_quality_ok(face):
            continue

        emb = getattr(face, "embedding", None)
        if emb is None:
            continue

        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        emb = emb / norm

        candidates.append(emb)
        print(f"[INFO] Captured {len(candidates)}/{target}")

    # Keep only the most consistent samples (removes outliers / bad frames).
    embs = np.stack(candidates, axis=0)
    mean = np.mean(embs, axis=0)
    mnorm = np.linalg.norm(mean)
    if mnorm != 0:
        mean = mean / mnorm
    sims = embs @ mean
    k = min(int(config.NUM_ENROLL_EMBEDDINGS), int(embs.shape[0]))
    keep_idx = np.argsort(sims)[-k:]

    for idx in keep_idx:
        db.add_embedding(name, embs[idx])

    user_total = db.count_user_embeddings(name)
    total = db.count_embeddings()
    db.close()
    cap.release()
    cv2.destroyAllWindows()
    safe_name = _safe_text(name)
    print(f"[SUCCESS] {safe_name} enrolled")
    print(f"[INFO] Embeddings stored for {safe_name}: {user_total}")
    print(f"[INFO] Total embeddings in DB: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name to enroll")
    parser.add_argument("--replace", action="store_true", help="Replace this user's embeddings (delete old + re-enroll)")
    parser.add_argument("--list-users", action="store_true", help="List users in the DB and exit")
    parser.add_argument("--delete-name", help="Delete all embeddings for this user and exit")
    parser.add_argument(
        "--delete-testusers",
        action="store_true",
        help="Delete all embeddings for usernames starting with 'test' and exit",
    )
    args = parser.parse_args()

    if args.list_users:
        db = FaceDatabase(config.DB_PATH)
        print(f"[INFO] DB Path: {_safe_text(_db_path_display())}")
        rows = db.list_users()
        if not rows:
            print("[INFO] No users found")
        else:
            for username, n in rows:
                print(f"{_safe_text(str(username))}\t{n}")
        db.close()
        raise SystemExit(0)

    if args.delete_testusers:
        db = FaceDatabase(config.DB_PATH)
        deleted = db.delete_users_like("test%")
        db.close()
        print(f"[INFO] Deleted {deleted} embeddings for usernames LIKE 'test%'")
        raise SystemExit(0)

    if args.delete_name:
        db = FaceDatabase(config.DB_PATH)
        deleted = db.delete_user(args.delete_name)
        db.close()
        print(f"[INFO] Deleted {deleted} embeddings for user: {_safe_text(args.delete_name)}")
        raise SystemExit(0)

    name = (args.name or "").strip()
    while not name:
        name = input("Enter the name to enroll: ").strip()

    enroll_from_camera(name, replace_existing=bool(args.replace))
