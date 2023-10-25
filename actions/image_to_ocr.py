import pandas as pd
import easyocr
from EasyOcrImageToTextGradio.utils.helper import draw_boxes


def inference(img, lang):
    reader = easyocr.Reader(lang, gpu=True)
    bounds = reader.readtext(img)
    im = img.copy()
    draw_boxes(im, bounds)
    return [im, pd.DataFrame(bounds, columns=['Coordinates', 'Text', 'Confidence']).iloc[:, 0:].astype(str)]
