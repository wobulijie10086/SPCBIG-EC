# -*- coding: utf-8 -*-
"""
@Time    : 2021/12/20 9:08
@Author  : LY
@FileName: gru.py
@SoftWare: PyCharm
"""
import keras.layers

from Parser import parameter_parser
from keras.layers import Input, ReLU, Dropout
from keras.layers.recurrent import GRU
from keras.models import Model

args = parameter_parser()


def get_gru(INPUT_SIZE, TIME_STEPS, dropout=args.dropout):
    """
    获得GRU最基础模型
    :param INPUT_SIZE: 输入的长度
    :param TIME_STEPS: 输入的宽度
    :param dropout: 损失率
    :return: GRU模型
    """
    inp = Input((INPUT_SIZE, TIME_STEPS))
    models = GRU(
        units=300,
        input_shape=(INPUT_SIZE, TIME_STEPS)
    )(inp)
    models = ReLU()(models)
    models = Dropout(dropout)(models)
    models = ReLU()(models)
    model = Model(inp, models)
    return model


def get_model():
    n_classes = 2
    inp = Input(shape=(100, 300))
    reshape = keras.layers.Reshape((1, 100, 300))(inp)
    #  pre=ZeroPadding2D(padding=(1, 1))(reshape)
    # 1
    conv1 = keras.layers.Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform')(reshape)
    # model.add(Activation('relu'))
    l1 = keras.layers.LeakyReLU(alpha=0.33)(conv1)

    conv2 = keras.layers.ZeroPadding2D(padding=(1, 1))(l1)
    conv2 = keras.layers.Convolution2D(32, 3, 3, border_mode='same', init='glorot_uniform')(conv2)
    # model.add(Activation('relu'))
    l2 = keras.layers.LeakyReLU(alpha=0.33)(conv2)

    m2 = keras.layers.MaxPooling2D((3, 3), strides=(3, 3))(l2)
    d2 = Dropout(0.6)(m2)
    # 2
    conv3 = keras.layers.ZeroPadding2D(padding=(1, 1))(d2)
    conv3 = keras.layers.Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform')(conv3)
    # model.add(Activation('relu'))
    l3 = keras.layers.LeakyReLU(alpha=0.33)(conv3)

    conv4 = keras.layers.ZeroPadding2D(padding=(1, 1))(l3)
    conv4 = keras.layers.Convolution2D(64, 3, 3, border_mode='same', init='glorot_uniform')(conv4)
    # model.add(Activation('relu'))
    l4 = keras.layers.LeakyReLU(alpha=0.33)(conv4)

    m4 = keras.layers.MaxPooling2D((3, 3), strides=(3, 3))(l4)
    d4 = Dropout(0.6)(m4)
    # 3
    conv5 = keras.layers.ZeroPadding2D(padding=(1, 1))(d4)
    conv5 = keras.layers.Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform')(conv5)
    # model.add(Activation('relu'))
    l5 = keras.layers.LeakyReLU(alpha=0.33)(conv5)

    conv6 = keras.layers.ZeroPadding2D(padding=(1, 1))(l5)
    conv6 = keras.layers.Convolution2D(128, 3, 3, border_mode='same', init='glorot_uniform')(conv6)
    # model.add(Activation('relu'))
    l6 = keras.layers.LeakyReLU(alpha=0.33)(conv6)

    m6 = keras.layers.MaxPooling2D((3, 3), strides=(3, 3))(l6)
    d6 = Dropout(0.5)(m6)
    # 4
    conv7 = keras.layers.ZeroPadding2D(padding=(1, 1))(d6)
    conv7 = keras.layers.Convolution2D(256, 3, 3, border_mode='same', init='glorot_uniform')(conv7)
    # model.add(Activation('relu'))
    l7 = keras.layers.LeakyReLU(alpha=0.33)(conv7)

    conv8 = keras.layers.ZeroPadding2D(padding=(1, 1))(l7)
    conv8 = keras.layers.Convolution2D(256, 3, 3, border_mode='same', init='glorot_uniform')(conv8)
    # model.add(Activation('relu'))
    l8 = keras.layers.LeakyReLU(alpha=0.33)(conv8)
    g = keras.layers.GlobalMaxPooling2D()(l8)
    print("g=", g)
    # g1=Flatten()(g)
    lstm1 = keras.layers.LSTM(
        input_shape=(100, 300),
        output_dim=256,
        activation='tanh',
        return_sequences=False)(inp)
    dl1 = Dropout(0.5)(lstm1)

    den1 = keras.layers.Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.6)(den1)

    #   lstm2=LSTM(
    #     256,activation='tanh',
    #     return_sequences=False)(lstm1)
    #   dl2=Dropout(0.5)(lstm2)
    print("dl2=", dl1)
    g2 = keras.layers.concatenate([g, dl2], axis=1)
    d10 = keras.layers.Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = keras.layers.LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.6)(l10)
    l11 = keras.layers.Dense(n_classes, activation='softmax')(l10)

    model = Model(input=inp, outputs=l11)
    model.summary()
    # 编译model
    adam = keras.optimizers.Adam(lr=0.003, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
