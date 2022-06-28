### round1 
模型结构：CascadeRCNN
GPU : 双卡 A40 
##### 有效trick
+ mixup
+ 后处理
+ 多尺度训练，大尺度训练
+ AutoAugment v1、albumentation
+ 4conv1fc

##### 下表缩写说明
aug = ShiftScaleRotate + RandomBrightnessContrast + RGBShift + HueSaturationValue
后处理的关键点：1.单张图片只有一种logo，可以利用预测结果进行最大置信度类别过滤；2. 测试集leak;

| method                                                                                          | 线下  | 线上                 |
|-------------------------------------------------------------------------------------------------|-----|--------------------|
| swin_base                                                                                       | --  | 0.551960           |
| swin_base + aug                                                                                 | -   | 0.557870           |
| swin_base + aug+切片SAHI                                                                          | -   | 0.534805           |
| swin_base + img_scale:(2560, 1000), (2560, 1750) + aug +FP16                                    | -   | 0.563815 |
| swin_base + (2560, 1000), (2560, 1750) + aug +FP16 + post-propocess+18epoch                     | -   | 0.616763           |
| swin_base + (2560, 1000), (2560, 1750) + aug +FP16 + post-propocess+18epoch+大尺度inference        | -   | 0.607210           |
| swin_base + (2560, 1000), (2560, 1750) + aug +FP16 + post-propocess+18epoch+rotato 10+5000      | -   | 0.606617           |
| swin_base + (2560, 1000), (2560, 1750) + aug +FP16 + post-propocess +SWA(4)                     | -   | 0.616752           |
| swin_base + (2560, 1000), (2560, 1750) + aug +FP16 + post-propocess +SWA(4)+max_per_img:5000    | -   | 0.616998           |
| swin_base + (3200, 1750), (3200, 2134) + aug +FP16 + post-propocess                             | -   | 0.611764 /0.611397 |
| swin_base + (2560, 1000), (2560, 1750) + autoaug +FP16 +  post-process                          | -   | 0.636606           |
| swin_base + (2560, 1000), (2560, 1750) + autoaug +FP16 +  post-process + gc_context             | -   | 0.623618           |
| swin_base + (2560, 1000), (2560, 1750) + autoaug +FP16 +  post-process  +mixup                  | -   | 0.654454           |
| swin_base + (2560, 1000), (2560, 1750) + autoaug +FP16 +  post-process  +mixup +albu            | -   | 0.657827           |
| CBnet     + (2560, 1000), (2560, 1750) + autoaug +FP16 +  post-process  +mixup +albu            | -   | 0.673100           |
| CBnet     + (2560, 1000), (2560, 1750) + autoaug +FP16 +  post-process  +mixup +albu  +4conv1fc | -   | 0.675344           |


### round2
将round1的checkpoint作为预训练模型，在round2的数据集上进行微调， 相较于round1 增加了更多的数据增强，包括IAASharpen、IAAEmboss，TTA增加了2个尺度

| method               | 线下  | 线上             |
|----------------------|-----|----------------|
| cbnet (finetune)     | --  | 0.592754/0.605 |
| cbnet  (wo finetune) | --  | 0.566647       |




| method                                 | 线下    | 线上        |
|----------------------------------------|-------|-----------|
| cbnet baseline                         | 0.479 |           |
| cbnet baseline  +mixup                 | 0.522 |           |
| cbnet baseline +Mixup+Albu             | 0.534 |           |
| cbnet baseline +Mixup+Albu+gn          | 0.522 |           |
| finetune +Mixup+Albu                   | 0.575 | 0.605377  |
| finetune +Mixup+Albu  +0.65 threshold  | 0.575 | 0.605274  |
| finetune +Mixup+Albu +more aug         | 0.578 | 0.606219  |
| finetune +Mixup+Albu +more aug + TTA 6 | 0.578 | 0.608623  |

### round1 训练命令
```

 bash ./tools/dist_train.sh  ./myconfig/round1.py  2


bash tools/dist_test.sh \
    ./myconfig/round1.py \
    ./work_dirs/round1/CBnet_4conv1fc/epoch_36.pth \
     4 \
     --format-only\
     --options "jsonfile_prefix=./results/round1"
     
```
需要对预测结果json进行后处理，首先运行modiftResults.py，然后将修改后的json路径输入并运行round1_myPostProcess.ipynb

### round2 训练命令
round2 以round1的checkpoint作为预训练模型，需要修改 配置文件的load_from
```

 bash ./tools/dist_train.sh  ./myconfig/round2.py  2



bash tools/dist_test.sh \
    ./myconfig/round2.py \
    ./work_dirs/round2/CBnet_4conv1fc/epoch_12.pth \
     4 \
     --format-only\
     --options "jsonfile_prefix=./results/round2"
     
```

需要对预测结果json进行后处理，只需要修改round2_myPostProcess.ipynb内的路径，并运行