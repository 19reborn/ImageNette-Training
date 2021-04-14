#	ImageNette  Training  Report

> A training experiment of ImageNette using ResNet in Pytorch

## Experimental Setup

### Platform

- GPU :  GTX1080Ti
- CUDA :  9.0.176
- Linux ：Ubuntu 16.04.5
- Python :  3.7.4
- Pytorch :  1.1.0

### Dataset

- [Imagenette](https://github.com/fastai/imagenette) :  a subset of 10 easily classified classes from Imagenet 

### Model

- **Net**

  使用Pytorch内置的预训练模型ResNet101，并将其中的FC层调整为2048*10，使得网络能对ImageNette中的10类标签进行分类.

- **Loss Function**

  使用交叉熵（CrossEntropyLoss）作为网络的损失函数。

- **Optimizer**

  使用Ranger（a synergistic optimizer combining RAdam (Rectified Adam) and LookAhead, and now GC (gradient centralization) in one optimizer）[^1]作为Optimizer。

- **Learning Rate Decay**

  使用余弦衰减函数（CosineAnnealingLR）在训练过程中对学习率进行调整，函数在Epoch=0时学习率最大，在总共训练的Epoch数+20处衰减为0。

## Result

- 选取初始的学习率为0.01，Batch为64，总共训练300个Epoch，训练结果如下：



<center class='half'>	
<img src=".\Result Image\loss_ResNet101_Batch=64,Epoch=300.png" alt="loss_batch=64,epoch=300"width="250" /><img src=".\Result Image\acc_ResNet101_Batch=64,Epoch=300.png" alt="acc_batch=64,epoch=300" width="250" />
</center>




> 左图表示训练集上的Loss和测试集上的Loss在训练过程中的变化曲线，右图表示训练集上的准确率和测试集上的准确率在训练过程中的变化曲线，横坐标为训练的Epoch数，纵坐标为相应的值。

由于采取了多种数据增强的方法，初始时训练集上的准确率大于测试集，随着训练的进行，在Epoch到150左右时，训练集上的准确率超过了测试集。训练了150个Epoch过后，测试集上的准确率增长缓慢，渐进到收敛值，而训练集上的准确率仍在不断提升。Epoch到300时，已经开始出现过拟合现象。

<img src=".\Result Image\lr_ResNet101_Batch=64,Epoch=300.png" alt="lr_batch=64,epoch=300" style="zoom: 40%;" />

同时我们还发现，在Epoch=30到100的训练过程中，测试集的loss和准确率有较大波动，说明此时学习率设置过大。考虑到我们的学习率衰减是余弦函数衰减（学习率变化曲线见上图），说明应适当减少总共训练的Epoch，从而增大学习率的衰减速率。

当Epoch=292时，测试集上的准确率最高，达到了94.75%；此时训练集上的准确率为97.49%。

- 选取初始的学习率为0.01，Batch为64，总共训练200个Epoch，训练结果如下：

<center class='half'>	
<img src=".\Result Image\loss_ResNet101_Batch=64,Epoch=200.png" alt="loss_batch=64,epoch=200"width="250" /><img src=".\Result Image\acc_ResNet101_Batch=64,Epoch=200.png" alt="acc_batch=64,epoch=300" width="250" />
</center>




总共训练的Epoch减少到200个后，Loss和准确率的曲线的平滑性有了较大提升。

当Epoch=173时，测试集上的准确率最高，达到了94.11%；此时训练集上的准确率为0.9517%。可见，测试集的准确率损失很少，但过拟合问题得到了较好的解决。

- 选取的初始学习率为0.01，Batch为32，总共训练200个Epoch，训练结果如下：

<center class='half'>	
<img src=".\Result Image\loss_ResNet101_Batch=32,Epoch=200.png" alt="loss_batch=32,epoch=200"width="250" /><img src=".\Result Image\acc_ResNet101_Batch=32,Epoch=200.png" alt="acc_batch=32,epoch=200" width="250" />
</center>


当使用更小的Batch Size后，发现在训练集上的Loss增加了3倍左右。在训练集上的数据增强引入了一定的随机性，这对标签的判断影响不大（即准确率受到的影响不大），但会给网路给出的概率分布添加噪声。当Batch Size较小时，噪声无法较好地抵消，从而计算出的交叉熵（Loss）会明显地增大。

## Analysis

### Epoch

设定初始学习率为0.01，Batch为64，改变训练的Epoch数，训练结束时的测试准确率结果如下：

​	

| Epoch | Test Accuracy(%) |
| :---: | :--------------: |
|  150  |      93.32       |
|  200  |      93.63       |
|  250  |      94.06       |
|  300  |      94.42       |
|  400  |      94.19       |

<img src=".\Result Image\Test Accuracy of different Epoch.png" alt="Test Accuracy of different Epoch" style="zoom:40%;" />

随着训练的Epoch数增加，测试准确率不断提升，说明模型仍有提升的空间。观察上文中Epoch=200和300的训练曲线，发现当训练的Epoch数较大时，测试准确率提升的速度已经变得相当缓慢了。Epoch=400时的测试准确率反而下降了，说明发生了过拟合现象。由于训练较大的Epoch单次花费的时间较多（训练300个Epoch需要5个小时左右），故在后续实验中选择Epoch=200。

### Batch Size

设定初始学习率为0.01，训练的Epoch数为200，改变Batch Size，训练过程中最优的测试准确率结果如下：

| Batch | Test Accuracy(%) |
| :---: | :--------------: |
|  32   |      93.53       |
|  40   |      94.29       |
|  48   |      93.91       |
|  56   |      93.53       |
|  64   |      94.11       |
|  72   |      94.24       |

<img src=".\Result Image\Test Accuracy of different Batch size.png" alt="Test Accuracy of different batch size" style="zoom:40%;" />

取Batch=40时，测试准确率达到了94.29%，似乎这是个相当不错的选择。但观察Batch=40时的训练曲线（如下图），会发现其相当不平稳，甚至出现了一次极大地波动。这是因为分类的类别较多，在Batch Size较小时，梯度更新的随机性较大，训练中容易出现震荡。

<center class='half'>	
<img src=".\Result Image\loss_ResNet101_Batch=40,Epoch=200.png" alt="loss_batch=40,epoch=200"width="250" /><img src=".\Result Image\acc_ResNet101_Batch=40,Epoch=200.png" alt="acc_batch=40,epoch=300" width="250" />
</center>



取Batch=56时，测试准确率变得很低，其训练曲线如下。曲线震荡较大，训练不够平稳。有趣的是， 在训练到100个Epoch之前，每过20个Epoch就会周期性地出现一次较大的波动。

<center class='half'>	
<img src=".\Result Image\loss_ResNet101_Batch=56,Epoch=200.png" alt="loss_batch=40,epoch=200"width="250" /><img src=".\Result Image\acc_ResNet101_Batch=56,Epoch=200.png" alt="acc_batch=40,epoch=300" width="250" />
</center>



取Batch=72时，测试准确率也达到了94.24%，似乎也是一个不错的选择。而其训练曲线同样存在一定的问题，即train loss始终大于test loss。这意味着需要训练更多的Epoch，只训练200个Epoch对于Batch=72时是不够的。

<center class='half'>	
<img src=".\Result Image\loss_ResNet101_Batch=72,Epoch=200.png" alt="loss_batch=72,epoch=200"width="250" /><img src=".\Result Image\acc_ResNet101_Batch=72,Epoch=200.png" alt="acc_batch=72,epoch=300" width="250" />
</center>



相对而言，在确定Epoch=200时，选取Batch=64在各方面（测试准确率，训练曲线平稳性）都有不错的表现。

### Model Depth

改变ResNet的深度，保持初始学习率为0.01，Batch Size为64，Epoch为200，依次测试ResNet18,ResNet34,ResNet50，得到的测试准确率如下（ResNet101使用之前训练的结果）：

|   Model   | Test Accuracy(%) |
| :-------: | :--------------: |
| ResNet18  |      93.58       |
| ResNet34  |      93.94       |
| ResNet50  |      94.22       |
| ResNet101 |      94.11       |

网络的深度增加到50层后，测试准确率不再上升，说明ResNet50的参数量已够胜任ImageNette的分类任务。ResNet50的训练曲线如下，相比相同参数设置下的ResNet101，ResNet50的损失函数整体的值更小，训练曲线也更为平稳。在刚开始进行实验时，为防止欠拟合的发生，选择了参数量更多的ResNet101完成了调参任务。但考虑到ResNet50不仅性能更好，训练时间也更短，本实验在网络架构选取上更优的选择应是ResNet50。

<center class='half'>	
<img src=".\Result Image\loss_ResNet50_Batch=64,Epoch=200.png" alt="loss_ResNet50_batch=64,epoch=200"width="250" /><img src=".\Result Image\acc_ResNet50_Batch=64,Epoch=200.png" alt="loss_ResNet50_batch=64,epoch=200" width="250" />
</center>



[^1]:https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

