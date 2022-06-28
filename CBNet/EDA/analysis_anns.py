import json
import random
import colorsys
import matplotlib.pyplot as plt
from datetime import datetime
import os
import cv2
import numpy as np
import shutil


class_num = 50
classes = ['冰墩墩', 'Sanyo/三洋', 'Eifini/伊芙丽', 'PSALTER/诗篇', 'Beaster', 'ON/昂跑', 'BYREDO/柏芮朵', 'Ubras', 'Eternelle', 'PERFECT DIARY/完美日记', '花西子', 'Clarins/娇韵诗', "L'occitane/欧舒丹", 'Versace/范思哲', 'Mizuno/美津浓', 'Lining/李宁', 'DOUBLE STAR/双星', 'YONEX/尤尼克斯', 'Tory Burch/汤丽柏琦', 'Gucci/古驰', 'Louis Vuitton/路易威登', 'CARTELO/卡帝乐鳄鱼', 'JORDAN', 'KENZO', 'UNDEFEATED', 'BOY LONDON', 'TREYO/雀友', 'carhartt', '洁柔', 'Blancpain/宝珀', 'GXG', '乐町', 'Diadora/迪亚多纳', 'TUCANO/啄木鸟', 'Loewe', 'Granite Gear', 'DESCENTE/迪桑特', 'OSPREY', 'Swatch/斯沃琪', 'erke/鸿星尔克', 'Massimo Dutti', 'PINKO', 'PALLADIUM', 'origins/悦木之源', 'Trendiano', '音儿', 'Monster Guardians', '敷尔佳', 'IPSA/茵芙莎', 'Schwarzkopf/施华蔻']
class2id = { c : idx for idx, c in enumerate(classes) }

f = json.load(open('./dataset/train/annotations/instances_train2017.json','r',encoding="utf-8"))
num = [0 for i in range(100)]
small = [0 for i in range(100)]
h = [0 for i in range(4000)]
w = [0 for i in range(4000)]
for image in f['images']:
    h[image['id']] = image['height']
    w[image['id']] = image['width']

    # myset[i] = set()
for image in f['annotations']:
    if(image['bbox'][2]*image['bbox'][3]<=75*75):
        small[image['category_id']] += 1
    num[image['category_id']] += 1
for i in range(1, 51):
    print(classes[i-1], num[i], small[i])


