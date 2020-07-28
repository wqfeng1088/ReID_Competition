# ReID
 
该项目是“武大锅盔队”在首届全国人工智能大赛（行人重试别赛道）初赛A轮和B轮使用的全部代码，用于审核组指导老师审查，万分感谢。

## 解题思路
我们使用罗浩等人在CVPR019发布的 Bag of Tricks and a Strong Baseline for Deep Person Re-Identification 作为我们的基础模型（以下简称reid strong baseline），同时，我们将ResNet50 [ Deep Residual Learning for Image Recognition ]的backbone换成了se_resnext101 [ Squeeze-and-Excitation Networks ]，并使用其在ImageNet图像分类挑战赛数据集上的预训练模型来获得进一步的性能提升。除了backbone之外，reid strong baseline中还包含了6个在reid社区中经常使用的tricks，我们也延续这些tricks的使用，分别是：

Warm Up
BNNeck
Label Smooth
Last Stride
Random Erasing
Center Loss

其中，warm up指的是在前10个epoch，学习率从线性增加到，然后在第30个epoch和第120个epoch时分别衰减10倍，最终在第150个epoch时训练结束；BNNeck是指在backbone的layer4输出后经过GAP（global average pool），得到的feature map，然后经过reshape得到全局特征，我们基于该特征计算triplet loss [ FaceNet: A Unifed Embedding for Face Recognition and Clustering ]和center loss [ A Discriminative Feature Learning Approach for Deep Face Recognition ]，之后这个特征依次经过batch norm [ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift ]层和分类层，得到预测的分类概率进行softmax loss计算。Label Smooth是对原图像进行标签平滑的操作，即将原始数据集上标注的one-hot形式标签修改为下面的形式：



## 1.项目下载和数据集路径准备
SeNet在ImageNet上的预训练模型：http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth

下载本项目：
`git clone https://github.com/wqfeng1088/ReID.git`

切换到下面目录：
`cd ReID/train_and_test`

修改其中torchreid.data.datamanager.py下的MyDataset类的`self.train_dir`，`self.val_query_dir`，`self.val_gallery_dir`，`self.query_dir`，`self.gallery_dir`为对应训练集，验证集query，gallery和测试集query，gallery目录。

## 2.一次完整的训练和测试
修改`MyDataset.py`中的`save_dir`为模型保存地址，`torchreid.data.datamanager.MyDataManager`中的`root`参数为数据集所在大目录(如`os.path.join( root, self.train_dir)`即为训练集完整目录)。然后运行`MyDataset.py`，即

`python3 MyDataset.py`


我们共训练6次模型，即执行上述操作六次。我们训练好的模型放在百度云中，链接为：
https://pan.baidu.com/s/1ut4ZguCexG2YJSsjsVaxQQ

（可选）如果需要执行聚类，只需要修改数据集目录，然后执行：`python3 kmeans.py`或`python3 cluster.py`进行KMeans聚类和MNNPL聚类。

## 2.多模型集成测试
切换到多模型集成测试目录：`cd ../test_jicheng`
类似地修改torchreid.data.datamanager.py下的MyDataset类的下的相关路径，然后修改`jicheng.py`下的`save_dir`为json文件保存地址，`torchreid.data.datamanager.MyDataManager`中的`root`参数为数据集所在大目录以及`fpath_1`~`fpath6`为训练好的6个模型所在目录，然后运行jicheng.py即可进行测试：

`python3 jicheng.py`

生成的json文件保存在`save_dir`下，我们生成的结果放在下面百度云中：
https://pan.baidu.com/s/1ut4ZguCexG2YJSsjsVaxQQ





感谢您的审阅。

