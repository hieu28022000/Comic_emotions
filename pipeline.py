import glob2
import cv2
import os 

import numpy as pd

from utils import get_config, loadImage, sorting_bounding_box, visual

from libs.CRAFT.craft import CRAFT
from libs.MORAN.MORAN_pred import MORAN_predict
from libs.MORAN.models.moran import MORAN
from libs.DeepText.Deeptext_pred import Deeptext_predict

from src import craft_text_detect

# setup config
cfg = get_config()
cfg.merge_from_file('configs/pipeline.yaml')
cfg.merge_from_file('configs/craft.yaml')

CRAFT_CONFIG = cfg.CRAFT
NET_CRAFT = CRAFT()

def text_recog(cfg, image_path):
    text = 'None'
    if cfg.PIPELINE.DEEPTEXT:
        list_image_path = [image_path]
        for img in list_image_path:
            text = Deeptext_predict(cfg.PIPELINE.DEEPTEXT_MODEL_PATH, img)
    elif cfg.PIPELINE.MORAN:
        text = MORAN_predict(cfg.PIPELINE.MORAN_MODEL_PATH, image_path, MORAN)
    
    return text

def text_detect_CRAFT(image_path, craft_config, net_craft, sortbb=True, visual_img=True):
    img = loadImage(image_path)
    bboxes, polys, score_text = craft_text_detect(img, craft_config, net_craft)

    if sortbb:
        polys = sorting_bounding_box(polys)
    if visual_img:
        img = visual(img, polys)
    
    return img, bboxes, polys, score_text

if __name__ == '__main__':
    # text = text_recog(cfg, 'data/illusion.png')
    img, bboxes, polys, score_text = text_detect_CRAFT('data/4_48_4.jpg', CRAFT_CONFIG, NET_CRAFT)
    for poly in polys:
        print(poly)
    cv2.imshow("image", img)
    cv2.waitKey(0)