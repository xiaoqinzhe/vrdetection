# vrdetection

## Paper
[AVR: Attention based Salient Visual Relationship Detection](https://arxiv.org/abs/2003.07012)

[学位论文：自然场景视觉关系检测算法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202101&filename=1020334585.nh&v=udehcmzL7Fd0FYgTeYyhV%25mmd2B%25mmd2BRpDvP07C7TeUEkxYN5CJHBRzkoM2Q6a%25mmd2FuYevkamYp)

## Project

### Dataset
VRD, VG数据集可在对应论文找到下载地址

### Prepare
代码基于[danfeiX/scene-graph-TF-release](https://github.com/danfeiX/scene-graph-TF-release)做的修改，一些准备工作可参考它的readme

### Train
运行脚本./experiments/scripts/train.sh，指定参数，会执行python ./tools/train_net.py
lib/networks里有各种网络，包括一些别的论文，和自己尝试过的网络。最终使用的网络是 **weightednet**

### Test
脚本为./experiments/scripts/test.sh，也需要指定相应参数

### tf_faster_rcnn
faster_rcnn的tf实现，可参考里面的readme，然后在对应数据集上训练，得到检测模型。
