import cv2
import  glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
f = glob.glob('../dataset/val/images/*.jpg')

w_list = []
h_list = []
for i, path in enumerate(f):
    print(i)
    pic = np.array(Image.open(path).convert('RGB'))
    w_list.append(pic.shape[0])
    h_list.append(pic.shape[1])


f, ax = plt.subplots(1,3, figsize=(16,4))
sns.histplot(w_list, ax=ax[0], palette=sns.light_palette("seagreen", as_cmap=True)).set_title('Width');
sns.histplot(h_list, ax=ax[1], palette=sns.color_palette("RdPu", 10)).set_title('Height');
sns.histplot(np.array(w_list)/np.array(h_list), ax=ax[2], palette=sns.color_palette("RdPu", 10)).set_title('W&H Ratio');
plt.show()