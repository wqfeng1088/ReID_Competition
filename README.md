# ReID
 
该项目是“武大锅盔队”在首届全国人工智能大赛（行人重试别赛道）初赛A轮和B轮使用的全部代码，用于审核组指导老师审查，万分感谢。

## 解题思路
我们使用罗浩等人在CVPR019发布的 《Bag of Tricks and a Strong Baseline for Deep Person Re-Identification 》作为我们的基础模型（以下简称reid strong baseline），同时，我们将ResNet50 《Deep Residual Learning for Image Recognition》的backbone换成了se_resnext101 《Squeeze-and-Excitation Networks》，并使用其在ImageNet图像分类挑战赛数据集上的预训练模型来获得进一步的性能提升。除了backbone之外，reid strong baseline中还包含了6个在reid社区中经常使用的tricks，我们也延续这些tricks的使用，分别是：
Warm Up
BNNeck
Label Smooth
Last Stride
Random Erasing
Center Loss

其中，warm up指的是在前10个epoch，学习率从线性增加到，然后在第30个epoch和第120个epoch时分别衰减10倍，最终在第150个epoch时训练结束；BNNeck是指在backbone的layer4输出后经过GAP（global average pool），得到的feature map，然后经过reshape得到全局特征，我们基于该特征计算triplet loss 《FaceNet: A Unifed Embedding for Face Recognition and Clustering》和center loss 《 A Discriminative Feature Learning Approach for Deep Face Recognition》，之后这个特征依次经过batch norm 《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》层和分类层，得到预测的分类概率进行softmax loss计算。Label Smooth是对原图像进行标签平滑的操作。

### 参数设置
我们的模型使用softmax loss和triplet loss联调，两者的损失权重均设置为1。而center loss的量级较大，为尽量保持损失在同一量级，我们将center loss的权重设置为0.0005，而rank loss的权重设置为2，于是总体损失为：
Loss = λ1 * Lid + λ2 * Ltri + λ3 * Lrank + λ4 * Lcenter
其中λ1=λ2=1,λ3=2,λ4=0.0005 而center loss使用单独的优化器进行优化，这一部分用到的所有参数设置均和罗浩等人的reid-strong-baseline保持一致。我们使用的代码是在KaiyangZhou的基础上修改的，并使用到了0.5概率的随机水平镜像，随机裁剪(先将图像放大，然后裁剪到 )以及概率为0.5的随机擦除作为数据增强。我们的初始学习率是，然后在前10个epochs线性增加到，随后保持不变，并在第30个epoch和第120个epoch时分别衰减10倍，直到第150个epoch时模型收敛，训练结束。我们同时训练6个相同的模型。

### 扩展（可选）
进行模型集成之前，其实我们还尝试了其他算法，简记如下： 我们根据训练好的模型(单模)对测试集B榜数据进行特征提取，根据提取的特征使用KMeans进行聚类(类别超参数设置为4000，大致为测试集A榜的3倍)，然后对聚类得到的结果打伪标签，并人工剔除其中明显错误的结果，得到的一个新的“训练集”，最后将该“训练集”和原始训练集合并作为最终训练集，对模型进行训练。当然，由于B轮时间限制，我们人工筛选数据后仍存在大量错误聚类结果，导致了最终结果反而略低于直接用原始训练集训练的结果(错误标签将模型带偏了)。此外，我们还尝试了将聚类得到的“训练集”在原始训练集上训练得到的模型上fine-tuning(这也是目前跨域reid的一个方向)，但同样由于B轮时间不够而无法进行有效调参，最终结果和原始模型相近，而这个过程所需时间却明显高于训练一个原始模型所需时间，于是在B轮有限的时间内只能放弃了这一做法。最后，除了KMeans聚类，我们还使用了我们组最近投稿到CVPR2020的一个聚类的工作，其思想是基于K互近邻和近邻可传递特点进行聚类，但同样受到时间限制而搁置。最终，我们还是选择时间最短，性能折衷的直接在原始训练集上训练模型，并同时重复训练6次，对结果进行融合。

### 运行环境
Python3.6: conda create -n py36 python=3.6 anaconda
pytorch：pip install torch==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

### 计算力
GPU：Tesla V100 1块
CPU：Intel Xeon E5-2630 v3


## 项目介绍
项目使用的预训练模型是在ImageNet2017图像分类比赛数据集上训练的se_resnext101_32x4d模型，模型所在地址为：预训练模型
首先使用cd命令修改到train_and_test目录下

### 数据集准备
根据官网发布的训练集，我们先在原始训练集中随机抽取500个ID作为本地验证集，其中每个ID选择一张图像作为验证集的query，剩余图像作为验证集的gallery。然后将剩余训练集中图像数少于3张的图像直接放到验证集gallery中作为干扰者，并将验证集中(含query和gallery)中图像数多余(含等于)3张图像的类别拷贝回训练集中作为最终训练集。对于以上数据，全部利用官网提供的txt文件进行重命名，格式为“0005_197997937.png”，其中0005是类别ID，197997937.png是原始图像名，而测试集不动。

其中torchreid.data.datamanager.py下的MyDataset类的`self.train_dir`，`self.val_query_dir`，`self.val_gallery_dir`，`self.query_dir`，`self.gallery_dir`为对应训练集，验证集query，gallery和测试集query，gallery目录。



然后将compute_mean_std.py中的train_path，test_query_path和test_gallery_path分别修改为训练集，测试集query和测试集gallery所在目录，并执行文件即可得到数据集的均值和方差。 运行程序示例：
python3 compute_mean_std.py

最终计算得到的数据集均值和方差分别为：
[0.09721232, 0.18305508, 0.21273703]和[0.17512791, 0.16554857, 0.22157137]。

### 模型训练

将MyDataset.py文件下的第15行中的save_dir修改为需要保存模型等输出文件的路径，第21行MyDataManager函数参数root修改数据所在路径。其他参数使用文件中的默认参数即可， 然后运行该文件即可开始训练模型。最后，我们额外训练该模型5次，保存在不同路径下。 运行程序示例：

python3 MyDataset.py

我们已经训练好的6个模型的百度云盘地址为： https://pan.baidu.com/s/1ut4ZguCexG2YJSsjsVaxQQ


### 聚类（可选）
将kmeans.py文件中save_dir修改为需要保存模型等输出文件的路径，MyDataManager函数的root参数修改为数据所在路径，fpath修改为在原训练集上训练得到的模型文件地址，KMeans函数中的n_clusters参数修改为需要聚类的类数，output_dir修改为聚类之后的图片保存地址，test_data_path修改为测试集B的query和gallery两者合并之后的文件路径，其余参数默认即可，然后执行该文件。 运行程序示例：

python3 kmeans.py

我们上传的代码支持在“ 扩展(可选)”部分提到的所有内容，由于提交时间限制，这里就不一一展开，均是类似的操作。简要说明如下：如果需要执行fine-tuning，则调用 MyDataManager_FineTune 类而不是 MyDataManager，并在对应的类内修改数据的路径并加载原训练模型即可；如果需要执行我们提出的基于K互近邻聚类算法，运行 cluster.py 文件并修改其中相应路径即可。
对于聚类结果，在时间运行的情况下，我们可以进行人工筛选。对于在测试集B上的聚类结果，我们考虑可以进行两种用途，要么与原始训练集合并，重头开始对模型训练，要么在原训练模型的基础上进行fine-tuning。
然后利用cd命令切换到test_jicheng目录下

### 测试

修改 jicheng.py 中的fpath1到fpath6为6个在原训练集上训练好的模型路径，save_dir为要保存输出结果的路径。 运行示例代码：

python3 jicheng.py

在save_dir下会输出日志文件和生成的预测json文件，其中json文件即为最终提交结果。 我们提交的json文件结果的链接为：https://pan.baidu.com/s/1ut4ZguCexG2YJSsjsVaxQQ



以上是本次ReID比赛的解题思路和步骤，用于评审，感谢评审老师的审阅。
