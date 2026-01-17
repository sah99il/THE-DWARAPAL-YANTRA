import cv2
from services.id_enrollment_service import enroll_from_id_image

# Replace with path to any ID photo image
IMAGE_PATH = "sample_id.jpg"

img = cv2.imread(IMAGE_PATH)

result = enroll_from_id_image(img, "ID_User")

print(result)