# Deepsort
#### 初赛：首先明确题意：多目标跟踪；指标MOTA和MOTP, 后期的大量实验证明检测算法相对于跟踪更重要。
#### 数据集分析：
##### 1.人群密集稀疏场景；
##### 2.场景（白天，黑夜）
##### 3.光照变化丰富。
##### 4.多方向视角，方向变化大；
##### 5.行人速度有快又慢。
## Config

Detection：

    Cascade-RCNN(HRNet) 基于mmdetection框架。
    采用多尺度训练（1216,608）和（1024,2048）, 多尺度测试：（1216,608），（1632,816）（2048，1024）
    常见数据增强crop 翻转，pad等
    丢帧后处理线性平滑
    修正框小于1==1
    多epoch平均的SWA

#### B榜 新增2个挑战： 更密集的人群和遮挡

#### 初赛不看速度要求，选择SOTA检测算法，Cascade-RCNN ，其中选择HRNet作为backbone。
##### Reid 模型 尝试了Deepsort自带的 类似于Resnet18, 后更换ResNet50ibn-a效果一般，发现涨分点不在这里。

# 中兴捧月阿尔法赛道决赛方案
## 开辟虚拟内存，控制物理内存
  python dlcmc.py 
## 程序执行
  python main.py
参数： --data_path  测试图片路径
	   --result_path  输出路径
### 
感谢好基友[何新](https://github.com/whut2962575697),多次彻夜讨论！
