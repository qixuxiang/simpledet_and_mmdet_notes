# simpledet_and_mmdet_notes
simpledet和mmdetection源码阅读笔


## 一.源码结构
### simpledet源码结构

```

├── config  #配置文件，定义网络结构
│   ├── cascade_r101v2_c5_red_1x.py
│   ├── ...
│   ├── __init__.py
│   ├── mask_r50v1_fpn_1x.py
│   ├── retina_r101v1_fpn_1x.py  #内含focal loss使用范例
│   ├── retina_r50v1_fpn_1x.py   #内含focal loss使用范例
│   ├── tridentnet_r101v2c4_c5_1x.py
│   ├── tridentnet_r101v2c4_c5_addminival_2x.py
│   ├── tridentnet_r101v2c4_c5_fastapprox_1x.py
│   ├── tridentnet_r101v2c4_c5_multiscale_addminival_3x_fp16.py
│   └── tridentnet_r50v2c4_c5_1x.py
├── core
│   ├── detection_input.py
│   ├── detection_metric.py
│   ├── detection_module.py
│   ├── __init__.py
├── detection_test.py  #测试infer1
├── detection_train.py #训练文件
├── mask_test.py   #测试infer2
├── models   #模型结构定义
│   └── tridentnet  #以tridentnet为例
│       ├── builder.py
│       ├── __init__.py
│       ├── input.py
│       ├── README.md
│       ├── resnet_v2_for_paper.py
│       └── resnet_v2.py
├── MODEL_ZOO.md
├── operator_cxx
├── operator_py
├── README.md
├── symbol  #网络RPN和box分类
│   ├── builder.py
│   ├── __init__.py
├── unittest #单元测试
├── utils
│   ├── callback.py
│   ├── contrib  #额外数据转换脚本
│   ├── generate_roidb.py  #生成训练测试文件
│   ├── __init__.py
│   ├── load_model.py
│   ├── logger.py
│   ├── lr_scheduler.py
│   ├── memonger_v2.py
└── voc
    └── README.md

```

### mmdetection源码结构

## 训练自己数据

无论是simpledet还是mmdetection，都是对coco数据优先支持。所以在开始之前建议把自己数据修改为标准的coco格式，各种类型数据转coco格式脚本见：[转换工具箱](https://github.com/spytensor/prepare_detection_dataset)。

### mmdetection训练自己数据

1.第一步当然是定义数据种类，需要修改的地方在`mmdet\datasets`。

在这个目录下新建一个文件，例如:`my_data.py`，然后把`coco.py`的内容复制过来，修改class类名为`MyDataset`最后把`CLASSES`的那个tuple改为自己数据集对应的种类tuple即可，例如：

```    
CLASSES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')
```
表明你的数据集有21类（包含背景），命名是'1'-'20'，还有一类背景。

2.接着在`mmdet\datasets\__init__.py`引入你自定义的数据集

在这个py文件中开头加入
```
from .my_data.py import MyDataset
```
然后在`__all__`列表中加入你的`MyDataset`

3.然后在`mmdet\core\evaluation\class_names.py`中加入你刚才的数据集类别

在开头加入
```
def mydata_classes():
    return [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'
    ]
```

然后在`dataset_aliases`字典中加入：
```
'mydata':['mydata']
```

最后在`mmdet\core\evaluation\__init__.py`引入你自定义的数据集，在`__all__`列表中加入你的`mydata_classes`。

4.最后在你训练的config文件中引入自己数据路径即可

在`config`文件中把`dataset_type`改为`mydata`，然后把`data_root`改成你coco格式自定义数据集的根目录即可。

### mmdetection激活所有模型的 Focal Loss

Focal Loss是何凯明在来retinenet中为了消除样本的类别不均衡，是在比赛中提分的利器。

在`mmdet/models/anchor_heads/anchor_head.py`的`AnchorHead`类中第44行`use_focal_loss`设置为True即可激活所有模型的 Focal Loss。注意：Focal loss在目标检测中仅在RPN阶段使用Focal Loss，因为用于预测GT的anchor box还是很少。

然后在config训练文件中的`train_cfg`的`rpn`最后加入：

```
smoothl1_beta=0.11,
gamma=2.0,
alpha=0.25,
allowed_border=-1,
pos_weight=-1,
debug=False
```
### mmdetection中cascade_mask_rcnn去掉语义分割分支改为cascade_rcnn

首先在`model`的dict中删去`mask_roi_extractor`和`mask_head`字段及其附属内容，接着在`train_cfg`的dict中删除所有的`mask_size=28`，最后在`data`中把所有的`with_mask=True`改为`with_mask=False`即可。
