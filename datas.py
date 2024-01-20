#检查错误图片
from PIL import Image
import conf
import matplotlib.image as mpi

#图像可用性检测
img_path=conf.train_imgs_path
w_p=conf.width
h_p=conf.hight
def check_useable(img):
    try:
        Image.open(img)

    except UserWarning as e:
        print(img)
        print('error!')

def cheak_shape(img):
    p=mpi.imread(img)
    h,w,c=p.shape
    if h < h_p:
        print(f'图片:{img}大小错误!')
    if w < w_p:
        print(f'图片:{img}大小错误!')

    if c != 3 :
        print(f'图片:{img}通道错误!')




if __name__ == '__main__':
    len=conf.train_imgs_len
    for i in range(len):
        img_name = img_path + str(i)+'.jpeg'
        check_useable(img_name)
        cheak_shape(img_name)
    print('ok')


