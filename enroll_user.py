import cv2
import numpy as np
from utils.aligner import align_face
from core.face_extractor import FaceExtractor
from database.db_manager import DatabaseManager

def main():
    # Initialize database manager
    db_manager = DatabaseManager(db_path="database/dwarapal.db")
    face_extractor = FaceExtractor() # Instantiate FaceExtractor

    # Get user's name
    user_name = input("Enter your name: ")
    if not user_name:
        print("Name cannot be empty.")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Look at the camera and press 'c' to capture.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow("Webcam - Press 'c' to capture", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Align face
            aligned_face = align_face(frame)

            if aligned_face is None:
                print("No face detected or confidence too low. Please try again.")
                continue

            # Save aligned face for verification
            cv2.imwrite(f"data/aligned/{user_name}.jpg", aligned_face)
            print(f"Aligned face saved to data/aligned/{user_name}.jpg")

            # Extract embedding
            embedding = face_extractor.get_embedding(aligned_face)

            if embedding is None:
                print("Could not extract embedding. Please try again.")
                continue

            # Save to database
            db_manager.add_user(user_name, embedding)
            print(f"User '{user_name}' enrolled successfully!")
            break

        elif key == ord('q'):
            print("Enrollment cancelled.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
