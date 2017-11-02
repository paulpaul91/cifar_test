import os
import shutil
import resnet
import densenet
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np

train_dir = 'train'
test_dir = 'test'
data_dir = '../cifar'
input_dir = 'train_valid_test'

input_str = data_dir + '/' + input_dir + '/'
batch_size = 128
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
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,
                                     transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1,
                                     transform=transform_test)
loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
print("ok for test")

import mxnet as mx
ctx=mx.gpu(0)
num_outputs = 10

net1 = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)
net1.hybridize()   
net1.load_params('densenet1.params', ctx=ctx)
net2 = resnet.ResNet164_v2(num_outputs)
net2.hybridize()   
net2.load_params('resnet1.params', ctx=ctx)


import pandas as pd

preds = []
for data, _ in test_data:
    output1 = nd.softmax(net1(data.as_in_context(ctx)))
    output2 = nd.softmax(net2(data.as_in_context(ctx)))
    output = nd.concat(*[output1, output2], dim=1)
    pred_label = output.argmax(1) % 10
    preds.extend(pred_label.astype(int).asnumpy())
    #print("ok")

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_ds.synsets[x])
df.to_csv('submissiontest.csv', index=False)
