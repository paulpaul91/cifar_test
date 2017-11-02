train_dir = 'train_tiny'
test_dir = 'test_tiny'
batch_size = 1

data_dir = '../data/kaggle_cifar10'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1
import os
import shutil

from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np

def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]),
                        brightness=0, contrast=0,
                        saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,
                                     transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1,
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid',
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1,
                                     transform=transform_test)
print("______do augment________")
loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
print("_____ok")
# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

from mxnet.gluon import nn
from mxnet import nd
import densenet
def get_net(ctx):
    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
import datetime
import sys
sys.path.append('..')
import utils
def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    filename = "mlp.params"


    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        # i=1
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
            net.save_params(filename)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            # print("once %f" % i)
            # i=i+1
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        if((valid_data is not None) and (valid_acc>0.05)):
            break
        if((valid_data is None) and ((train_acc / len(train_data))>0.05)):
            break
ctx = utils.try_gpu()
num_epochs = 200
learning_rate = 0.1
weight_decay = 5e-4
lr_period = 50
lr_decay = 0.1
import numpy as np
import pandas as pd
net = get_net(ctx)
net.hybridize()
print("shushihua")
train(net, train_data, valid_data, num_epochs, learning_rate,
      weight_decay, ctx, lr_period, lr_decay)
filename1 = "mlp1.params"
net.save_params(filename1)
print("do vaild test")
print("do vaild test")
preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submissi111on11.csv', index=False)


print("full train")
# net = get_net(ctx)
# net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate,
      weight_decay, ctx, lr_period, lr_decay)
print("do full test")
preds1 = []
for data, label in test_data:
    output1 = net(data.as_in_context(ctx))
    preds1.extend(output1.argmax(axis=1).astype(int).asnumpy())

sorted_ids1 = list(range(1, len(test_ds) + 1))
sorted_ids1.sort(key = lambda x:str(x))

df1 = pd.DataFrame({'id': sorted_ids1, 'label': preds1})
df1['label'] = df1['label'].apply(lambda x: train_valid_ds.synsets[x])
df1.to_csv('submissi111on111.csv', index=False)