import cv2
from utils.aligner import align_face
from core.face_extractor import get_embedding
from core.matcher import FaceMatcher
from security.rppg_liveness import LivenessDetector
from database.db_manager import DatabaseManager

def start_dwarapal():
    cap = cv2.VideoCapture(0)
    db = DatabaseManager()
    matcher = FaceMatcher(threshold=0.65)
    liveness = LivenessDetector()
    
    print("Dwarapal System Active. Scanning...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Check Liveness first (The Shield)
        is_live, bpm = liveness.check_liveness(frame)
        
        # 2. Process Identity (The Brain)
        aligned = align_face(frame)
        status_text = "Scanning..."
        color = (255, 255, 255)

        if aligned is not None:
            embedding = get_embedding(aligned)
            users = db.fetch_all_users()
            
            for user in users:
                result = matcher.verify(user['embedding'], embedding)
                if result['match'] and is_live:
                    status_text = f"Verified: {user['name']} | BPM: {bpm}"
                    color = (0, 255, 0) # Green for Success
                    break
                elif result['match'] and not is_live:
                    status_text = "SPOOF DETECTED (No Pulse)"
                    color = (0, 0, 255) # Red for Attack
        
        # UI Overlay
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Dwarapal - AI Biometric Gatekeeper", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_dwarapal()