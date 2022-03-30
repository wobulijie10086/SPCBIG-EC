# -*- coding: utf-8 -*-
"""
@Time    : 2022/01/22 9:00
@Author  : LY
@FileName: scnn.py
@SoftWare: PyCharm
"""
from Parser import parameter_parser
from keras.layers import Input, ReLU, Dropout,Concatenate,Dense, Activation,LeakyReLU,Flatten,Bidirectional
from keras.layers.recurrent import GRU,LSTM,SimpleRNN
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.optimizers import adam


args = parameter_parser()





def model2():
    inp2 = Input(shape=(100, 300))


    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                   activation='relu',)(inp2)
    print(conv1)
    l1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters=100, kernel_size=2, strides=1, padding='same',
                      activation='relu', )(l1)
    l2 = Activation('relu')(conv2)
    x2 = Dense(300, activation='relu')(l2)
    bigru = Bidirectional(GRU(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False), merge_mode='concat')(x2)
    dl1 = Dropout(0.5)(bigru)

    d10 = Dense(1024)(dl1)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)
    # l11 = Flatten()(l10)
    l12 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp2], outputs=l12)

    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model2_gru():
    inp2 = Input(shape=(100, 300))


    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                   activation='relu',)(inp2)
    print(conv1)
    l1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters=100, kernel_size=2, strides=1, padding='same',
                      activation='relu', )(l1)
    l2 = Activation('relu')(conv2)
    x2 = Dense(300, activation='relu')(l2)
    gru = GRU(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x2)
    dl1 = Dropout(0.3)(gru)

    d10 = Dense(1024)(dl1)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)
    # l11 = Flatten()(l10)
    l12 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp2], outputs=l12)

    model.summary()
    # 编译model
    adam1 = adam(lr=0.003, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model2_blstm():
    inp2 = Input(shape=(100, 300))


    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                   activation='relu',)(inp2)
    print(conv1)
    l1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters=100, kernel_size=2, strides=1, padding='same',
                      activation='relu', )(l1)
    l2 = Activation('relu')(conv2)
    x2 = Dense(300, activation='relu')(l2)
    blstm = Bidirectional(LSTM(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False),merge_mode='concat')(x2)
    dl1 = Dropout(0.5)(blstm)

    d10 = Dense(1024)(dl1)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.7)(l10)
    # l11 = Flatten()(l10)
    l12 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp2], outputs=l12)

    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model2_lstm():
    inp2 = Input(shape=(100, 300))


    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                   activation='relu',)(inp2)
    print(conv1)
    l1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters=100, kernel_size=2, strides=1, padding='same',
                      activation='relu', )(l1)
    l2 = Activation('relu')(conv2)
    x2 = Dense(300, activation='relu')(l2)
    lstm = LSTM(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x2)
    dl1 = Dropout(0.3)(lstm)

    d10 = Dense(1024)(dl1)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.3)(l10)
    # l11 = Flatten()(l10)
    l12 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp2], outputs=l12)

    model.summary()
    # 编译model
    adam1 = adam(lr=0.003, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model

def model2_rnn():
    inp2 = Input(shape=(100, 300))


    conv1 = Conv1D(filters=300, kernel_size=2, strides=2, padding='same', input_shape=(100, 300),
                   activation='relu',)(inp2)
    print(conv1)
    l1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters=100, kernel_size=2, strides=1, padding='same',
                      activation='relu', )(l1)
    l2 = Activation('relu')(conv2)
    x2 = Dense(300, activation='relu')(l2)
    rnn = SimpleRNN(
        # input_shape=(100, 300),
        output_dim=150,
        activation='tanh',
        return_sequences=False)(x2)
    dl1 = Dropout(0.5)(rnn)

    d10 = Dense(1024)(dl1)
    # model.add(Activation('relu'))
    l10 = LeakyReLU(alpha=0.33)(d10)
    l10 = Dropout(0.5)(l10)
    # l11 = Flatten()(l10)
    l12 = Dense(2, activation='softmax')(l10)

    model = Model(inputs=[inp2], outputs=l12)

    model.summary()
    # 编译model
    adam1 = adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    # adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)

    # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
    model.compile(loss='categorical_crossentropy', optimizer=adam1, metrics=['accuracy'])
    return model
