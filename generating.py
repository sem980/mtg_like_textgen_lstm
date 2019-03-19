# -*- encoding: utf-8 -*-

from keras.models import load_model
from MorphTable import MorphTable
import numpy as np
import json
import sys
import warnings

def passing_warn(*args,**kwargs):
    pass

def lottery(y_pred):
    #次に予測される形態素の確率が与えられた際に抽選を行う関数
    #多項分布で抽選を行い，np.argmaxでインデックスを返す
    pred_f64 = np.asarray(y_pred).astype('float64')
    pred_norm = pred_f64/sum(pred_f64)
    hit_index = np.argmax(np.random.multinomial(1,pred_norm,1))
    return hit_index

def seed_gen(filename):
    #テキスト生成の際の最初の一形態素をランダムに出力
    with open(filename,'r') as f:
        seeds = json.load(f)
    return np.random.choice(seeds)

def generate_text(model,seed_morph,morph_table,maxlen=5,eps=0.1):
    #文頭の形態素seed_morphからmorph_tableに基づくインデックスのリストを作成
    seed_seq = ['<BOS>']*(maxlen-1)+[seed_morph]
    seq_index = [morph_table.morph2index(m) for m in seed_seq]
 
    #テキストの生成
    #<EOS>か形態素200個が出力されるまでテキストを生成
    generate_list = list()
    n = 0
    next_morph = ''
    while(n<200 and next_morph != '<EOS>'):
        #学習済みモデルへの入力となるx_predはseq_indexのOne-Hot表現
        x_pred = np.zeros((1,maxlen,morph_table.ret_typenum()),dtype=np.bool)
        for i,morph_index in enumerate(seq_index):
            x_pred[0,i,morph_index] = 1
        #x_predに基づき次の形態素を予測する確率y_predを取得
        y_pred = model.predict(x_pred,verbose=0)[0]

        #ランダムに得られたeps_choiceがeps以下の値であれば抽選
        #以上であれば最大の確率となる形態素を取得
        eps_choice = np.random.random()
        hit = np.argmax(y_pred) if eps_choice > eps else lottery(y_pred)
        next_morph = morph_table.index2morph(hit)

        #seq_indexの最初の形態素を捨て, 予測された形態素を最後に追加
        seq_index.pop(0)
        seq_index.append(hit)
        generate_list.append(next_morph)
        n+=1
    
    return ''.join(seed_seq[-1:]+generate_list[:-1])

def main():
    warnings.warn = passing_warn

    model = load_model('model/morphlevel_model.h5')
    #形態素とインデックスの変換用テーブル
    morph_table = MorphTable('data/vocab.json')
    # 1sequenceの形態素数
    maxlen = 5
    #epsの値によって予測時に抽選が行われる確率が変化
    if len(sys.argv)>1:
        eps = float(sys.argv[1])
    else:
        eps = 0.8

    seed_morph = seed_gen('data/seed_morph.json')
    gen = generate_text(model,seed_morph,morph_table,maxlen,eps)
    print(gen)

if __name__ == '__main__':
    main()