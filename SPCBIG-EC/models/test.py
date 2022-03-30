from __future__ import print_function

import tensorflow as tf
import warnings
import torch.nn
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, ReLU, GRU, Concatenate, Flatten
from keras.optimizers import Adamax
from keras.layers import Layer
from keras import backend as K
from keras import initializers, regularizers, constraints
from sklearn.model_selection import train_test_split
from Parser import parameter_parser
from models.loss_draw import LossHistory


from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

args = parameter_parser()

warnings.filterwarnings("ignore")

from torch.autograd import Variable
class cnn:
    conv1 = torch.nn.Conv1d(in_channels=300,out_channels = 300, kernel_size = 2, stride=1, padding=0)
    conv2 = torch.nn.Conv1d(in_channels=99,out_channels = 100, kernel_size = 2, stride=1, padding=1)
    input = torch.randn(64, 100, 300)
    # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    input = input.permute(0, 2, 1)
    input = Variable(input)
    out1 = conv1(input)
    out1 = out1.permute(0, 2, 1)
    out1 = Variable(out1)
    out2 = conv2(out1)
    res1 = out1.reshape(out1.size(1), -1)
    res2 = out2.view(out2.size(1), -1)
    #torch.stack((res1,res2),0).size()
    print(res1.shape)
    print(res2.shape)
    print(out1.size())
    print(out2.size())
