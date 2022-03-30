# -*- coding: utf-8 -*-
"""
@Time    : 2022/01/21 9:08
@Author  : LY
@FileName: spcnn.py
@SoftWare: PyCharm
"""
from Parser import parameter_parser
from keras.layers import Input, ReLU, Dropout,Concatenate,Dense, Activation,LeakyReLU,Bidirectional,Flatten
from keras.optimizers import adam
from keras.layers.recurrent import GRU,LSTM,RNN,SimpleRNN
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import numpy


args = parameter_parser()


def model1():
    inp = Input(shape=(100,300))

    # reshape = Reshape((11, 5, 1))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                      activation='relu')(inp)
    print(conv1)
    l1 = Activation('relu')(conv1)

    x1 = Dense(300, activation='relu')(l1)
    # model=Model(input=inp,outputs=x1)
    model = Model(input=inp, outputs=x1)
    return model
def model2():

    inp2 = Input(shape=(100, 300))

    # reshape = Reshape((11, 5, 1))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                   activation='relu',)(inp2)
    print(conv1)
    l1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters=100, kernel_size=2, strides=1, padding='same',
                      activation='relu', )(l1)
    l2 = Activation('relu')(conv2)
    x2 = Dense(300, activation='relu')(l2)
    model = Model(input=inp2, outputs=x2)
    return model


def merge_model():
    model_1 = model1()
    model_2 = model2()

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = Concatenate(axis=1)([r1, r2])
    print(x.shape)

    gru1 = Bidirectional(GRU(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False),merge_mode='concat')(x)
    dl1 = Dropout(0.8)(gru1)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.8)(den1)

    #   lstm2=LSTM(
    #     256,activation='tanh',
    #     return_sequences=False)(lstm1)
    #   dl2=Dropout(0.5)(lstm2)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.5)(d10)
    l10 = Dropout(0.8)(l10)

    # l11= Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp1, inp2], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def merge_model_gru():
    model_1 = model1()
    model_2 = model2()

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = Concatenate(axis=1)([r1, r2])
    print(x.shape)

    gru1 = GRU(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x)
    dl1 = Dropout(0.5)(gru1)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.5)(den1)

    #   lstm2=LSTM(
    #     256,activation='tanh',
    #     return_sequences=False)(lstm1)
    #   dl2=Dropout(0.5)(lstm2)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)

    # l11= Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp1, inp2], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def merge_model_blstm():
    model_1 = model1()
    model_2 = model2()

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = Concatenate(axis=1)([r1, r2])
    print(x.shape)

    blstm = Bidirectional(LSTM(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False),merge_mode='concat')(x)
    dl1 = Dropout(0.5)(blstm)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.5)(den1)

    #   lstm2=LSTM(
    #     256,activation='tanh',
    #     return_sequences=False)(lstm1)
    #   dl2=Dropout(0.5)(lstm2)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)

    # l11= Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp1, inp2], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def merge_model_lstm():
    model_1 = model1()
    model_2 = model2()

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = Concatenate(axis=1)([r1, r2])
    print(x.shape)

    lstm = LSTM(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x)
    dl1 = Dropout(0.5)(lstm)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.5)(den1)

    #   lstm2=LSTM(
    #     256,activation='tanh',
    #     return_sequences=False)(lstm1)
    #   dl2=Dropout(0.5)(lstm2)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)

    # l11= Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp1, inp2], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def merge_model_rnn():
    model_1 = model1()
    model_2 = model2()

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = Concatenate(axis=1)([r1, r2])
    print(x.shape)

    rnn = SimpleRNN(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x)
    dl1 = Dropout(0.5)(rnn)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.5)(den1)

    #   lstm2=LSTM(
    #     256,activation='tanh',
    #     return_sequences=False)(lstm1)
    #   dl2=Dropout(0.5)(lstm2)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.3)(l10)

    # l11= Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp1, inp2], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.003, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model