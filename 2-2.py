########### 导入数据库
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def main():
    ###数据模块###
    class COVID19Dataset(Dataset):
        def __init__(self,root_dir,txt_path,transform=None):
            """
            输入：
            （1）root_dir:图片路径
            （2）txt_path:标签路径
            （3）Transform:图像预处理集合
            输出：
            作用：获取数据集的路径、预处理的方法
            """
            self.root_dir=root_dir
            self.txt_path=txt_path
            self.transform=transform
            self.img_info=[] #[(path,label),..,]
            self.label_array=None
            self._get_img_info()

        def __getitem__(self, index):
            """
            作用：从磁盘中读取数据，并预处理
            :param item: 图像数字的索引
            :return: 图像数据+标签
            """
            path_img,label=self.img_info[index]
            img=Image.open(path_img).convert("L")

            if self.transform is not None:
                img=self.transform(img)

            return img,label

        def __len__(self):
            """
            作用：统计数据集的长长度
            :return: 数据集的长长度
            """
            if len(self.img_info)==0:
                raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                    self.root_dir))

            return len(self.img_info)

        def _get_img_info(self):
            """
            功能：实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list[(path,label),...]
            :return: [(path,label),...]
            """
            with open(self.txt_path,"r") as f:
                txt_data=f.read().strip()
                txt_data=txt_data.split("\n")

            self.img_info=[(os.path.join(self.root_dir,i.split()[0]),int(i.split()[2]))
                           for i in txt_data]

    root_dir=r"D:\AI\File\PyTorch-Tutorial-2nd-main\PyTorch-Tutorial-2nd-main\data\covid-19-demo"
    img_dir=os.path.join(root_dir,"imgs")
    path_txt_train=os.path.join(root_dir,"labels","train.txt")
    path_txt_valid=os.path.join(root_dir,"labels","valid.txt")
    transforms_func=transforms.Compose([
        transforms.Resize((8,8)),
        transforms.ToTensor(),
    ])

    trian_data=COVID19Dataset(root_dir=img_dir,txt_path=path_txt_train,
                              transform=transforms_func)
    valid_data=COVID19Dataset(root_dir=img_dir,txt_path=path_txt_valid,
                              transform=transforms_func)

    train_loader=DataLoader(dataset=trian_data,batch_size=2)
    valid_loader=DataLoader(dataset=valid_data,batch_size=2)

    #####模型模块###
    class TinnyCNN(nn.Module):
        def __init__(self,cls_num=2):
            super(TinnyCNN,self).__init__()
            self.conv=nn.Conv2d(1,1,kernel_size=(3,3))
            self.fc=nn.Linear(36,cls_num)

        def forward(self,x):
            x=self.conv(x)
            # 思考，跟reshape的区别
            x=x.view(x.size(0),-1)
            out=self.fc(x)
            return out

    model=TinnyCNN(2)

    #####优化模块
    # 分类损失函数：交叉熵
    loss_f=nn.CrossEntropyLoss()
    # 优化器：SGD
    # 思考：优化器中的动量和权重衰弱如何作用
    optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    # 学习率函数
    # 思考：如何使用，gamma如何使用
    scheduler=optim.lr_scheduler.StepLR(optimizer,gamma=0.1,step_size=50)

    #####迭代模块
    for epoch in range(1000):
        # 训练集训练
        for data,labels in train_loader:
            # forward & backward
            outputs=model(data)
            optimizer.zero_grad()

            loss=loss_f(outputs,labels)
            loss.backward()
            optimizer.step()

            # 计算分类准确率
            # 思考：torch.max的输出是什么
            _,predicted=torch.max(outputs.data,1)
            correct_num=(predicted==labels).sum()
            acc=correct_num/labels.shape[0]
            print("Epoch:{},Train Loss:{:.2f},Acc:{:.0%}".format(epoch,loss,acc))

            # 验证集验证
            model.eval()
            for data, label in valid_loader:
                # forward
                outputs = model(data)

                # loss 计算
                loss = loss_f(outputs, labels)

                # 计算分类准确率
                _, predicted = torch.max(outputs.data, 1)
                correct_num = (predicted == labels).sum()
                acc_valid = correct_num / labels.shape[0]
                print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc_valid))

            # 添加停止条件
            if acc_valid==1:
                print("超参数最优")
                break

            # 学习率调整
            scheduler.step()

if __name__=="__main__":
    main()







