import conf
import train as t
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
#读取测试图片路径
test_img_path='D:/test_imgs/h12.jpeg'

#test_img_size
h=conf.img_size
w=conf.img_size

#图片处理
test_img=transforms.Compose([
transforms.RandomCrop((w,h)),
    transforms.Resize((w,h)),
    transforms.ToTensor()
])
class Test_Datas(Dataset):
    def __init__(self,transforms = None):
        self.imgs=[]
        img=Image.open(test_img_path)
        self.imgs.append(img)
        self.transforms = transforms

    def __getitem__(self, index):
        img=self.imgs[index]
        p_img=self.transforms(img)
        return p_img

    def __len__(self):
        return len(self.imgs)

data=Test_Datas(test_img)
dataloder=DataLoader(dataset=data,batch_size=1)

if __name__ == '__main__':
    code=-1
    img=Image.open(test_img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('test_img')
    print('推测结果:')
    name_list=['雷电将军','甘雨','胡桃','刻晴','申鹤','宵宫','夜兰','神里绫华']
    for i in dataloder:
        code=t.predict(i)
        if code != -1 :
            print(name_list[code])
        else:
            print('无法推测!')

    plt.show()
