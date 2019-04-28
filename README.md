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
