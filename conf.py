import torch

device=torch.device('cpu')
batch_size=64
learning_rate=0.001 #推荐0.001
train_imgs_path='./train_x/'
train_labels_path='./train_y/labels.txt'
epoch=90
img_size=640
save='./mod/model9.pt'
train_imgs_len=429
test_model='./mod/model8.pt'