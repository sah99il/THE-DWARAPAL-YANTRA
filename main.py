import sys
import os

# Get the absolute path of the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))
# Add the project root to the Python path
sys.path.insert(0, project_root)

import cv2
import numpy as np # Import numpy for creating blank image
from utils.aligner import align_face
from database.db_manager import DatabaseManager
from core.face_extractor import FaceExtractor
from core.matcher import FaceMatcher
from security_dwarapal.rppg_liveness import LivenessDetector

def start_dwarapal():
    cap = cv2.VideoCapture(0)
    db = DatabaseManager()
    matcher = FaceMatcher(threshold=0.65)
    liveness = LivenessDetector()
    face_extractor = FaceExtractor() # Instantiate FaceExtractor
    
    # Model Warming: Dummy inference call
    print("Warming up models...")
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8) # Create a dummy image
    dummy_aligned = align_face(dummy_image)
    if dummy_aligned is not None:
        face_extractor.get_embedding(dummy_aligned)
    print("Models warmed up. Dwarapal System Active. Scanning...")

    frame_count = 0
    last_status_text = "Scanning..."
    last_color = (255, 255, 255)
    last_bpm = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1

        # 1. Check Liveness first (The Shield) - always run
        is_live, confidence, bpm = liveness.check_liveness(frame)
        
        # 2. Process Identity (The Brain) - only every 5th frame
        if frame_count % 5 == 0:
            aligned = align_face(frame)
            status_text = "Scanning..."
            color = (255, 255, 255) # Default white for scanning

            if aligned is not None:
                embedding = face_extractor.get_embedding(aligned)
                users = db.fetch_all_users()
                
                found_match = False
                for user in users:
                    result = matcher.verify(user['embedding'], embedding)
                    if result['match'] and is_live:
                        status_text = f"Verified: {user['name']}"
                        color = (0, 255, 0) # Green for Success
                        found_match = True
                        break
                    elif result['match'] and not is_live:
                        status_text = "SPOOF DETECTED (No Pulse)"
                        color = (0, 0, 255) # Red for Attack
                        found_match = True
                        break
                
                if not found_match and aligned is not None: # Face detected but no match
                    status_text = "Unrecognized Face"
                    color = (0, 165, 255) # Orange for unrecognized
            else: # No face detected in this frame
                status_text = "No Face Detected"
                color = (255, 255, 0) # Yellow for no face

            last_status_text = status_text
            last_color = color
            last_bpm = bpm
        
        # UI Overlay - always display the last known status
        # Ensure BPM is always displayed based on the liveness detection, not just every 5th frame
        display_status_text = f"{last_status_text} | Current BPM: {bpm:.1f}" if is_live else last_status_text
        cv2.putText(frame, display_status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
        cv2.imshow("Dwarapal - AI Biometric Gatekeeper", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_dwarapal()