# -*- encoding: utf-8 -*-
import json

#形態素⇔インデックス間の返還を行うクラス
class MorphTable:
    def __init__(self,filename):
        with open(filename,'r') as f:
            morph_list = json.load(f)
        self.dic_morph2index = dict((v,int(n)) for n,v in morph_list.items())
        self.dic_index2morph = dict((int(n),v) for n,v in morph_list.items())
        self.typenum = len(morph_list)
        
    def morph2index(self,morph):
        return self.dic_morph2index[morph]
    
    def index2morph(self,index):
        return self.dic_index2morph[index]
    
    def ret_typenum(self):
        return self.typenum