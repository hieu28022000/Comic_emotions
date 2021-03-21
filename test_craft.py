from skimage import io
import cv2
import numpy as np
from src import craft_text_detect
from utils import get_config
from libs.CRAFT.craft import CRAFT
from utils import sorting_bounding_box
from utils import visual

'''
Download weigth của craft trên drive về bỏ vào thư mục libs/CRAFT/models
Chạy file này, thay link ảnh đầu vào, đầu ra của file này sẽ trả về danh sách xmin, ymin, xmax, ymax của các bounding box
đã được sắp xếp theo thứ tự từ trái qua phải, từ trên xuống dưới
'''


def loadImage(img_file):
    img = cv2.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

# setup config
cfg = get_config()
cfg.merge_from_file('configs/craft.yaml')
craft_config = cfg.CRAFT

# run craft
net = CRAFT()
img = loadImage('data/test.jpg')
print ('--------craft processing----------')
bboxes, polys, score_text = craft_text_detect(img, craft_config, net)
polys = sorting_bounding_box(polys)
# hàm lưu ảnh kết quả xem chơi
visual(img, polys)
print ('--------craft done  ----------')
