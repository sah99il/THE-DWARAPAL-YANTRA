import cv2
from services.verification_service import verify_frame

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("🔴 Live Gatekeeper started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = verify_frame(frame)

    # Decide label
    if result["verdict"]:
        text = f"{result['user']} ✓ LIVE"
        color = (0, 255, 0)
    else:
        text = result.get("reason", "Processing...")
        color = (0, 0, 255)

    # Draw overlay
    cv2.putText(
        frame,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA
    )

    if "identity_score" in result:
        score_text = f"ID: {result['identity_score']:.2f} | LIVE: {result['liveness_score']:.2f}"
        cv2.putText(
            frame,
            score_text,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.imshow("Dwarapala Gatekeeper (Live)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()