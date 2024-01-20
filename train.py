import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import conf

#该页面用于图形训练
device=conf.device
batch_size=conf.batch_size #训练分组大小
learning_rate=conf.learning_rate

#图像预处理
#设定图像输入标准大小
hight=conf.img_size #高度
width=conf.img_size #宽度
train_imgs=transforms.Compose([
    transforms.RandomCrop((hight,width)),
    transforms.Resize((hight,width)),
    transforms.ToTensor()
])

test_imgs=transforms.Compose([
    transforms.Resize((hight,width)),
    transforms.ToTensor()
])
#读取数据
#D:/train1/train_imgs/
#D:/train1/labels.txt
train_imgs_path=conf.train_imgs_path
train_labels_path=conf.train_labels_path
save=conf.save
class Train_Datas(Dataset):
    def __init__(self,transforms = None):
        self.labels=[]
        fp=open(train_labels_path,'r',encoding='utf-8')
        text=fp.readlines()
        for i in text:
            list1=[]
            fi=int(i)
            list1.append(fi)
            self.labels.append(list1)

        self.imgs=[]
        for i in range(len(self.labels)):
            imgs_path = train_imgs_path+str(i)+'.jpeg'
            img=Image.open(imgs_path)
            self.imgs.append(img)

        self.transforms = transforms

        assert len(self.labels) == len(self.imgs),'number of imgs or labels error!'

    def __getitem__(self, index):
        img=self.imgs[index]
        label=self.labels[index]
        p_img=self.transforms(img)
        return p_img,label

    def __len__(self):
        return len(self.labels)

class Mymod(nn.Module):
    def __init__(self):
        super(Mymod,self).__init__()
        self.conv_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=20,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_3=nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_4=nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_5=nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_6=nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_8 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        #全连接层
        self.linear1=nn.Linear(2560,1000)
        self.linear2=nn.Linear(1000,512)
        self.linear3=nn.Linear(512,8)


    def forward(self,x,batch_size):
        z1=self.conv_1(x)
        z2=self.conv_2(z1)
        z3=self.conv_3(z2)
        z4=self.conv_4(z3)
        z5=self.conv_5(z4)
        z6=self.conv_6(z5)
        z7=self.conv_7(z6)
        z8=self.conv_8(z7)
        arr=z8.view(batch_size,2560)
        z9=self.linear1(arr)
        z9=F.relu(z9)
        z10=self.linear2(z9)
        z10=F.relu(z10)
        z11=self.linear3(z10)
        return z11

epoch=conf.epoch
def train():
    mod = Mymod()
    mod = mod.to(device)
    optim = Adam(mod.parameters(), lr=learning_rate)
    datas = Train_Datas(train_imgs)

    for i in range(1,epoch+1):
        #round=i%100
        data_loads=DataLoader(dataset=datas,batch_size=batch_size,shuffle=True,drop_last=True)
        if i==epoch:
            print(f'epoch:{i}')
        if i==1:
            print(f'epoch:{i}')
        for j,(x,label) in enumerate(data_loads):
            x=x.to(device)
            y=label[0]
            y=y.to(device)
            optim.zero_grad()
            pre=mod(x,batch_size=batch_size)
            pre1=F.log_softmax(pre,dim=1)
            loss=F.nll_loss(pre1,y)
            if i==epoch:
                print(f'loss:{loss}')
            if i==1:
                print(f'loss:{loss}')
            loss.backward()
            optim.step()
    #模型保存
    torch.save(mod.state_dict(),save)
    print('---训练完成---')

model=conf.test_model
def predict(test_x):
    mod = Mymod()
    mod.eval()
    mod = mod.to(device)
    test_x=test_x.to(device)
    mod.load_state_dict(torch.load(model))
    pre=mod(test_x,batch_size=1)
    pre1 = F.softmax(pre, dim=1)
    arr = pre1.cpu().data.numpy()
    list1 = arr.tolist()
    list2 = list1[0]
    k=-1
    sw=list2[0]
    for i in range(len(list2)):
        if list2[i] >= sw:
            sw=list2[i]
            k=i

    return k


if __name__ == '__main__':
    train()