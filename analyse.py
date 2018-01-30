
from __future__ import absolute_import  # 导入3.x的特征函数
from __future__ import print_function

import jieba
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.models import load_model
maxlen = 50

def train():
    negative = pd.read_csv('./data/tweet_negative5-2.csv', header=None, error_bad_lines=False)
    positive = pd.read_csv('./data/tweet_positive5-2.csv', header=None, error_bad_lines=False)
    negative['mark'] = 0
    positive['mark'] = 1

    pos_neg = pd.concat([positive, negative], ignore_index=True)
    print('cut words start')
    pos_neg['words'] = pos_neg[0].apply(lambda x: list(jieba.cut(x)))
    print('cut words end')
    d2v_train = pos_neg['words']
    total_words = []
    for w in d2v_train:
        total_words.extend(w)
    print('word count')
    word_count = pd.DataFrame(pd.Series(total_words).value_counts())
    del total_words, d2v_train
    word_count['id'] = list(range(1, len(word_count)+1))
    word_count.to_csv('./model/wc.csv', index=True, encoding='utf-8')
    print('sentence vector')
    pos_neg['sent'] = pos_neg['words'].apply(lambda x: list(word_count['id'][x]))

    print("Pad sequences (samples x time)")
    pos_neg['sent'] = list(sequence.pad_sequences(pos_neg['sent'], maxlen=maxlen))
    x = np.array(list(pos_neg['sent']))[::2]  # 训练集
    y = np.array(list(pos_neg['mark']))[::2]
    xt = np.array(list(pos_neg['sent']))[1::2]  # 测试集
    yt = np.array(list(pos_neg['mark']))[1::2]
    xa = np.array(list(pos_neg['sent']))  # 全集
    ya = np.array(list(pos_neg['mark']))

    print('Build model...')
    model = Sequential()
    model.add(Embedding(len(word_count) + 1, 256))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # try using a GRU instead, for fun
    # model.add(Dropout(0.5))
    # model.add(Dense(128))
    # model.add(Activation('sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(xa, ya, batch_size=32, nb_epoch=2, validation_data=(xt, yt))  # 训练时间为若干个小时
    score, acc = model.evaluate(xt, yt,
                                batch_size=32)
    model.save('model/lstm2.h5')
    print('Test score:', score)
    print('Test accuracy:', acc)

def cut(s, wc):
    words = list(jieba.cut(s))
    seq = []
    for word in words:
        if word in wc['id']:
            seq.append(wc['id'][word])
        else:
            seq.append(0)

    return seq

def predict(s):
    model = load_model('model/lstm.h5')
    wc = pd.read_csv('./model/wc.csv', index_col=0)

    seq = sequence.pad_sequences(cut(s, wc), maxlen=maxlen)
    print(seq)
    result = model.predict_classes(seq)
    print(result)

# predict('最最最最tm讨厌抽烟没有烟德的人了')
train()
# classes = model.predict_classes(xa)
# acc = np_utils.accuracy(classes, ya)
# print('Test accuracy:', acc)
