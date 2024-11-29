[ENGLISH](README.md) | 简体中文

# **MindCos**

- [MindCos介绍](#MindCos介绍)
- [应用案例](#应用案例)
    - [事例重建](#事例重建)
    - [喷注鉴别](#喷注鉴别)
    - [磁场预测](#磁场预测)
- [安装教程](#安装教程)

## **MindCos介绍**

粒子物理学和宇宙学是探索自然界最基本规律的科学领域，它们通过高能对撞机和粒子加速器等尖端设备产生的实验数据来揭示物质的基本结构和宇宙的起源。在这些领域中，数据分析是理解实验结果和验证理论模型的关键步骤。然而，这些实验产生的数据量巨大，且复杂性高，传统的数据处理方法往往难以应对。

为了解决这一挑战，我们旨在通过先进的算法和计算技术，提高数据分析的效率和准确性，通过深度学习模型，来识别和分析粒子碰撞事件、粒子轨迹和能量分布等复杂现象。

与传统的数据处理方法相比，我们的机器学习库能够显著降低计算成本，缩短数据处理时间，同时提高结果的准确性。它为粒子物理学和宇宙学的研究者提供了一个强大的工具，使他们能够更深入地探索物质的基本结构和宇宙的起源。

MindCos是基于昇思MindSpore开发的，它不仅支持多种实验数据的分析，还能够适应不同的研究需求。我们的目标是为广大的科研人员、高校教师和学生提供一个高效、易用的粒子物理和宇宙学数据分析软件，以推动粒子物理学和宇宙学的进一步发展。

## 应用案例

### 事例重建

|案例|数据集|模型架构|GPU|CPU|
|-----|-----|-----|-----|-----|
|径迹重建|缪子反常磁矩实验合成数据|图神经网络|✔|✔|
|中微子重建|海玲计划模拟数据|图神经网络|X|X|

### 喷注鉴别

|案例|数据集|模型架构|GPU|CPU|
|-----|-----|-----|-----|-----|
|ParticleNet|顶夸克数据集|EdgeConv|X|X|
|LLP@CEPC|CEPC模拟希格斯粒子数据|异构图神经网络|✔️|✔️|
|LorentzNet|夸克胶子喷注数据集|洛伦兹等变图神经网络|✔️|✔️|

### 磁场预测

|案例|数据集|模型架构|GPU|CPU|
|-----|-----|-----|-----|-----|
|磁场预测|亥姆霍兹线圈磁场数据|PINN|✔️|✔️|
|磁场预测|亥姆霍兹线圈磁场数据|DeepONet|X|X|

## 安装教程

依照教程安装MindSpore
```bash
https://www.mindspore.cn/install
```
例子：
```bash
conda create -n MindCos python==3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/x86_64/mindspore-2.2.14-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装依赖库
```bash
pip install -r requirements.txt
```

安装GPU版本的mindspore：[安装说明](gpu_version_install.txt)





