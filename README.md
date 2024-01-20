This task is about images recognition.

And aim is recognize roles from picture which is come from a game named Genshin Impact. It just came to me on a whim. Maybe it can used to other side.

I choose 8 roles. It mean that this is a multi-classification task.

I used convolutional neural networks, It is basic on VGG-A. But I made some adjustments. 

In this task, I used pytorch, because this makes it easier to build neural networks.

The names and numbers that correspond are as follows：

['Raiden Shogun','Ganyu','Hu Tao','Keqing','Shenhe','Yoimiya','Yelan','Kamisato Ayaka']

['0','1','2','3','4','5','6','7']

The input test image already has a sample in the test folder, corresponding to the **test_imgs**. When using the model, please change the **test_img_path** in **test_mod.py** to the corresponding name path.

The input image should be as close to the size of 640x640 as possible, so as not to crop the image correctly to the feature. Do not use a picture smaller than 640x640.

The trained model is located in the **./mod/** path.

The pictures should be preprocessed before training.
In **datas.py** there are ways to check the availability of the training set images to avoid errors.
Run **train.py** for training. Before training, please ensure that all the pictures correspond to the labels, and the size of the pictures should not be less than 640x640, as close as possible to 640x640.
Set the path to the training set image in **conf.py**. The image should be in jpeg format and named in order encoding.

Training equipment requirements：

RAM: >16GB

Video memory : >= 8GB

这是一个图像识别的项目。

这个项目的目的是从图片中识别出原神中的角色，这个是我的突发奇想，在其他方面可能会有一些作用。

我选择了8个角色，这是一个对图像的8分类任务。这个项目基于VGG-A模型，我在此基础上做出了一些改动，这个项目采用pytorch框架。

本案例使用CNN神经网络对640x*640x*3的图片进行识别和分类，共分为8类：
以下为名称以及编号对应：
['雷电将军','甘雨','胡桃','刻晴','申鹤','宵宫','夜兰','神里绫华']
['0','1','2','3','4','5','6','7']
输入的测试图片在测试文件夹中已经有样例，对应在test_imgs下，使用模型的时候请将test_mod.py中的test_img_path修改为对应的名称路径。
输入的图片请尽量接近640x*640的大小，以免图像裁切的时候不能正确裁切到特征，不要使用小于640*x640的图片。
训练好的模型位于./mod/路径下。
关于模型训练：
训练前需对图片进行预处理。
在datas.py中有对训练集图像进行检测可用性的方法，以避免出错。
运行train.py进行训练，训练前请确保所有的图片跟标签对应，并且图片大小不能小于640x*640，尽量接近640x*640。
在conf.py中设置好训练集图片的路径，图片要使用jpeg格式，并且采用按顺序编码的命名方式。
训练设备要求：
内存：>16GB
显存: >=8GB
