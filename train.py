# -*- encoding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense,LSTM,BatchNormalization
from keras.optimizers import RMSprop
from MorphTable import MorphTable
import numpy as np 
import json

def build_model(seq_size,num_class):
    #単層LSTM
    lstm = LSTM(units=128,input_shape=(seq_size,num_class))
    
    model = Sequential()
    model.add(lstm)
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(num_class,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.01))
    return model

def make_seq_vec(mtg_data,morph_table,maxlen=5,step=4):
    #学習用データの整形
    #最大形態素長maxlenのシーケンスとそれに対応する次の形態素をone-hotで表現
    sentences = list()
    next_morph = list()
    
    #データから学習用シーケンスを切り取り
    for v in mtg_data.values():
        text = ['<BOS>']*(maxlen-1)+v['analyzed']+['<EOS>']
        
        first_text = text[:2*maxlen-2]
        sentences.extend([first_text[i:i+maxlen] for i in range(0,len(first_text)-maxlen)])
        next_morph.extend([first_text[i] for i in range(maxlen,len(first_text))])
        
        last_text = text[maxlen-1:]
        sentences.extend([last_text[i:i+maxlen] for i in range(0,len(last_text)-maxlen,step)])
        next_morph.extend([last_text[i] for i in range(maxlen,len(last_text),step)])
    
    #データを学習のためのOne-hotに整形
    x = np.zeros((len(sentences),maxlen,morph_table.ret_typenum()),dtype=np.bool)
    y = np.zeros((len(next_morph),morph_table.ret_typenum()),dtype=np.bool)
    for i ,sentence in enumerate(sentences):
        for j,morph in enumerate(sentence):
            x[i,j,morph_table.morph2index(morph)] = 1
        y[i,morph_table.morph2index(next_morph[i])] = 1
    return x,y

def main():
    maxlen = 5
    epochs = 40
    with open('data/mtg_flavor.json','r') as f:
        mtg_data = json.load(f)
    morph_table = MorphTable('data/vocab.json')

    #学習データ作成
    x,y = make_seq_vec(mtg_data,morph_table)
    #モデルデータの作成
    model = build_model(maxlen,morph_table.ret_typenum())
    #学習
    model.fit(x,y,batch_size=128,epochs=epochs)
    #保存
    #model.save('morphlevel_model.h5',include_optimizer=False)

if __name__ == '__main__':
    main()