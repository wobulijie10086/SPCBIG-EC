# -*- coding: utf-8 -*-
"""
@Time    : 2022/01/22 9:00
@Author  : LY
@FileName: cnn.py
@SoftWare: PyCharm
"""
import keras.layers

from Parser import parameter_parser
from keras.layers import Input, ReLU, Dropout,Concatenate,Dense, Activation,LeakyReLU,Bidirectional,Flatten
from keras.optimizers import adam
from keras.layers.recurrent import GRU,LSTM,RNN,SimpleRNN
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D


args = parameter_parser()




def model1():
    inp = Input(shape=(100,300))

    # reshape = keras.layers.Reshape((1, 100, 300))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                      activation='relu')(inp)
    print(conv1)
    l1 = Activation('relu')(conv1)

    x1 = Dense(300, activation='relu')(l1)
    # model=Model(input=inp,outputs=x1)

    bigru = Bidirectional(GRU(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False), merge_mode='concat')(x1)
    dl1 = Dropout(0.5)(bigru)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.5)(den1)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)
    # l11 = Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model1_gru():
    inp = Input(shape=(100,300))

    # reshape = keras.layers.Reshape((1, 100, 300))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                      activation='relu')(inp)
    print(conv1)
    l1 = Activation('relu')(conv1)

    x1 = Dense(300, activation='relu')(l1)
    # model=Model(input=inp,outputs=x1)

    gru = GRU(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x1)
    dl1 = Dropout(0.2)(gru)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.2)(den1)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.3)(l10)
    # l11 = Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.003, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model1_blstm():
    inp = Input(shape=(100,300))

    # reshape = keras.layers.Reshape((1, 100, 300))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                      activation='relu')(inp)
    print(conv1)
    l1 = Activation('relu')(conv1)

    x1 = Dense(300, activation='relu')(l1)
    # model=Model(input=inp,outputs=x1)

    blstm = Bidirectional(LSTM(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False), merge_mode='concat')(x1)
    dl1 = Dropout(0.2)(blstm)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.2)(den1)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.2)(l10)
    # l11 = Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.002, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model1_lstm():
    inp = Input(shape=(100,300))

    # reshape = keras.layers.Reshape((1, 100, 300))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                      activation='relu')(inp)
    print(conv1)
    l1 = Activation('relu')(conv1)

    x1 = Dense(300, activation='relu')(l1)
    # model=Model(input=inp,outputs=x1)

    lstm = LSTM(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x1)
    dl1 = Dropout(0.2)(lstm)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.2)(den1)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.2)(l10)
    # l11 = Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.003, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model1_rnn():
    inp = Input(shape=(100,300))

    # reshape = keras.layers.Reshape((1, 100, 300))(inp)
    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                      activation='relu')(inp)
    print(conv1)
    l1 = Activation('relu')(conv1)

    x1 = Dense(300, activation='relu')(l1)
    # model=Model(input=inp,outputs=x1)

    rnn = SimpleRNN(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x1)
    dl1 = Dropout(0.5)(rnn)

    den1 = Dense(300, activation="relu")(dl1)
    # model.add(Activation('relu'))
    # l11=LeakyReLU(alpha=0.33)(d11)
    dl2 = Dropout(0.2)(den1)

    print("dl2=", dl1)
    g2 = Concatenate(axis=1)([dl1, dl2])
    d10 = Dense(1024)(g2)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)
    # l11 = Flatten()(l10)
    l11 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp], outputs=l11)
    model.summary()
    # 编译model
    adam1 = adam(lr=0.004, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

