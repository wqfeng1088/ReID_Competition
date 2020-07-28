# ReID
 
该项目是“武大锅盔队”在首届全国人工智能大赛（行人重试别赛道）初赛A轮和B轮使用的全部代码，用于审核组指导老师审查，万分感谢。

您可以在我们的KLab项目中获取项目的更详细的介绍，这里我们只简单介绍项目的训练和测试流程。

KLab项目链接：https://www.kesci.com/home/project/share/44a983872814a643

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

