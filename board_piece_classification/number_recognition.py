from PIL import Image
import pytesseract
import numpy as np
import cv2


def preprocess_image(img, zoom, threshold):
    w, h = img.size

    # First zoom to the center of the image
    x = w / 2
    y = h / 2

    zoom2 = zoom * 2

    img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))

    # Then keep the number only (so anything that is either black or bright red)
    # and convert the rest to white
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > threshold else 0)
    img = img.convert("1")

    img.show()

    return img


img_path = (
    "board_piece_classification/data/output/generated_synthetic_tiles/bg_0_brick_6.png"
)

img1 = np.array(preprocess_image(Image.open(img_path), 3.5, 85))
text = pytesseract.image_to_string(img1, config="--psm 3")

print(text)
