#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models ,datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from PIL import Image


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[3]:


from mxnet.gluon import data as gdata
train_transform = transforms.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为224*224的新图像
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0),ratio=(3.0/4.0, 4.0/3.0)),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机变化亮度、对比度和饱和度    
    transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    #缩放到256*256的图像
    transforms.Resize(256),
    #中心裁剪到224*224的图像
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[4]:


train_dataset = datasets.ImageFolder(
    "./data/imagenette2/train",
    train_transform
)

test_dataset =datasets.ImageFolder(
    "./data/imagenette2/val",
    test_transform
)


# In[5]:


Batch = 64
EPOCH = 200
LR = 0.01


# In[6]:


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=Batch, num_workers=5)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=Batch, num_workers=5)


# In[7]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet101(pretrained=False)
        self.model.fc = nn.Linear(2048, 10)
    
    def forward(self, x):
        output = self.model(x)
        return output


# In[8]:


import ranger

net=Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = ranger.Ranger(net.parameters(),lr=LR,eps=1e-6)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,EPOCH+20)

loss_history=[]
acc_history=[]
tacc_history=[]
tloss_history=[]
lr_list=[]
best_acc=0
best_epoch=0


# In[ ]:


start_time = time.time()
for epoch in range(EPOCH):
    epoch_time = time.time()
    epoch_loss = 0
    correct = 0
    total=0    
    scheduler.step()
    lr_list.append(scheduler.get_lr())
    print("Epoch {} / {}".format(epoch, EPOCH))
    net.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        outputs = net(inputs) # 前向传播
        loss = criterion(outputs, labels) # softmax + cross entropy
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() 
        epoch_loss += loss.item() 
        outputs = nn.functional.softmax(outputs,dim=1)
        _, pred = torch.max(outputs, dim=1)
        correct += (pred.cpu() == labels.cpu()).sum().item()
        total += labels.shape[0]
    acc = correct / total
    loss_history.append(epoch_loss/len(labels))
    acc_history.append(acc)
    
    #计算测试集准确率及Loss
    correct = 0
    total = 0
    test_loss = 0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)            
            outputs = net(images)
            loss = criterion(outputs, labels) # softmax + cross entropy
            test_loss += loss.item()
            outputs = nn.functional.softmax(outputs,dim=1)
            _, pred = torch.max(outputs, dim=1)
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.shape[0]
    
    tacc = correct / total
    
    epoch_time2 = time.time()
    print("Duration: {:.0f}s, Train Loss: {:.4f}, Train Acc: {:.4f}， Test Acc : {:.4f}, Test Loss: {:.4f}".format(epoch_time2-epoch_time, epoch_loss/len(labels), acc, tacc, test_loss/len(labels)))
    if tacc>best_acc:
        best_acc = tacc
        best_epoch = epoch
        torch.save(net.state_dict(), './model_1.pth') #保存模型
    end_time = time.time()
    
    tacc_history.append(tacc)
    tloss_history.append(test_loss/len(labels))
    
print("Total Time:{:.0f}s".format(end_time-start_time))


# In[ ]:


plt.plot(np.arange(1,EPOCH+1),lr_list,'b-',color = 'b')
plt.xlabel("epoch")#横坐标名字
plt.ylabel("Learning Rate")#纵坐标名字
plt.title('Learning Rate Curve')
plt.savefig('./nopre_result/lr_test.png', bbox_inches = 'tight')
plt.close


# In[ ]:


x=np.arange(1,EPOCH+1)
loss_history=np.array(loss_history)
tloss_history=np.array(tloss_history)
plt.plot(x,loss_history,'s-',color = 'r', label = 'train_loss')#s-:方形
plt.plot(x,tloss_history,'o-',color = 'g', label ='test_loss')#o-:圆形
plt.legend()
plt.xlabel("epoch")#横坐标名字
plt.ylabel("Loss")#纵坐标名字
plt.title('Loss Curve')
plt.savefig('./nopre_result/loss_test.png', bbox_inches = 'tight')
plt.close()


# In[ ]:


x=np.arange(1,EPOCH+1)
acc_history=np.array(acc_history)
tacc_history=np.array(tacc_history)
plt.plot(x,acc_history,'s-',color = 'r', label = 'train_accuracy')#s-:方形
plt.plot(x,tacc_history,'o-',color = 'g', label ='test_accuracy')#o-:圆形
plt.legend()
plt.xlabel("epoch")#横坐标名字
plt.ylabel("Accuracy")#纵坐标名字
plt.title('Accuracy Curve')
plt.savefig('./nopre_result/acc_test.png', bbox_inches = 'tight')
plt.close()


# In[ ]:


print('Best Accuracy of the network on the test images: %.4f %%' % (100*best_acc))
print('Best epoch: %d' % (best_epoch))


# In[ ]:





# In[ ]:




