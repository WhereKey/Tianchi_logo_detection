import json
import random
import colorsys
import matplotlib.pyplot as plt
from datetime import datetime
import os
import cv2
import numpy as np
import tensorflow as tf
import shutil


class_num = 50
classes = ['冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻']
hsv_tuples = [(1.0 * x / class_num, 1., 1.) for x in range(50)]
def draw_bbox_new(image, bboxes, show_label=False):
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / class_num, 1., 1.) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(32)
    random.shuffle(colors)
    random.seed(None)
    for i, bbox in enumerate(bboxes):
        # print(bbox)
        x,y,w,h = bbox[0][:4]
        class_ind = bbox[1]-1
        bbox_color = colors[class_ind]
        bbox_thick = 3
        c1 = (int(x), int(y))
        c2 = (int(x+w), int(y+h))
        # print(type(c1), c1)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], 1)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1 )[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return image


class pic:
    def __init__(self, id, path):
        self.image_id = id
        self.path = path
        self.bbox_list = []
    def add_bbox(self, bbox):

        self.bbox_list.append(bbox)

f = json.load(open('../dataset/train/annotations/instances_train2017.json','r',encoding="utf-8"))
print(f.keys())
pics = {}
# print(f['annotations'])
for image in f['images']:
    a = pic(image['id'], image['file_name'])
    pics[int(image['id'])] = a
# import numpy
a = set()
for image in f['annotations']:
    # pics[int(image['image_id'])].add_bbox([image['bbox'], image['category_id']])
    a.add( tuple([image['bbox'][2],image['bbox'][3]]))
print(a)




