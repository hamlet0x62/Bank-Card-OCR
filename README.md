# Bank-Card-OCR
A bank-card OCR system built with Flask, Vue and Tensorflow.
## 项目简介
该项目是一个基于深度学习的银行卡识别系统，是用于参加第九届`cnsoftbei`大赛的参赛作品。由于该届大赛已经结束，本项目也并未获得奖项，故现将其开源出来。

项目主要分为两个部分，一部分是使用Flask+Vue编写的Web App，另一部分是银行卡号识别模块。

其中，银行卡卡号识别模块具有两个功能：
- 卡号区域检测（检测模型来自对[eragonruan/text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)的再训练）
- 卡号识别（使用CNN + LSTM构建的图像序列识别模型）

银行卡识别模块中，使用[tensorflow/serving](https://github.com/tensorflow/serving)来为识别模型搭建HTTP服务，Web App的Server端通过HTTP请求访问银行卡识别模块的识别服务。


## Build Guide
本来已为该项目编写好了较为完备的`markdown`格式的 build 文档，但现已无法找到，只有已经编译好的 HTML 文件，点击[这里](https://hamlet0x62.github.io/Bank-Card-OCR/readme.html)查看。

## 注意
由于GitHub不允许上传大尺寸文件，在此仓库中只上传了已经训练好的卡号识别模型的检查点文件。

由于卡号区域检测模型的检查点文件过于庞大（>200M），故在此提供卡号区域检测模型的下载地址:[百度网盘地址](https://pan.baidu.com/s/14Czr0gh_mMkUGw8ij9l03A)（提取码：cjoq）。

卡号区域检测模型的所有检查点文件清单如下，请将其放入`/detection/model`文件夹下。
```bash
├── checkpoint
├── ctpn_55000.ckpt.data-00000-of-00001
├── ctpn_55000.ckpt.index
└── ctpn_55000.ckpt.meta
```

## 最后
若发现项目中存在值得改进的地方，还希望各位批评指正，共同进步。
