import random
import cv2
import numpy as np
from PIL import Image

def degrade_id_image(pil_img):
    img = np.array(pil_img)

    # Downscale + upscale
    scale = random.uniform(0.4, 0.7)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    img = cv2.resize(img, (w, h))

    # JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 60)]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(enc, 1)

    # Color desaturation
    if random.random() < 0.5:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return Image.fromarray(img)
