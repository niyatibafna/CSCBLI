import torch
from torch.utils.data import Dataset, DataLoader
import io
import numpy as np
import torch.nn as nn
import argparse
import pdb
from dictionary import Dictionary
from model import Discriminator
from preprocess import MyDataset
import json


parser = argparse.ArgumentParser(description='Discriminator testing')
parser.add_argument("--model_path", type=str, default='model.pkl', help="point model path")
parser.add_argument("--dict_path", type=str, default='', help="point test dic")
parser.add_argument("--src_lang", type=str, default='en', help="point src lang")
parser.add_argument("--tgt_lang", type=str, default='ar', help="point tgt lang")
parser.add_argument("--reload_src_ctx", type=str, default='', help="point tgt lang")
parser.add_argument("--reload_tgt_ctx", type=str, default='', help="point tgt lang")

parser.add_argument("--mode", type=str, default='v1', help="point tgt lang")
parser.add_argument("--lambda_w1", type=float, default=0, help="")

parser.add_argument("--output_path", type=str, default='', help="Write output dictionary to file")

params = parser.parse_args()


def test(model_path, dict_path, output_path):
    torch._use_new_zipfile_serialization = False
    model = torch.load(model_path, map_location=torch.device('cuda:0'))

    s_src_w2i = model.static_src_dico.word2id
    s_tgt_w2i = model.static_tgt_dico.word2id
    c_src_w2i = model.context_src_dico.word2id
    c_tgt_w2i = model.context_tgt_dico.word2id

    s2t_dict = dict()
    
    if params.reload_src_ctx !="" and params.reload_tgt_ctx !="":
        a1, a2, a3, a4 = model.reload_context(params.reload_src_ctx, params.reload_tgt_ctx, params.src_lang, params.tgt_lang)
        model.vecmap_context_src_dico, model.vecmap_context_src_embed, model.vecmap_context_tgt_dico, model.vecmap_context_tgt_embed = a1, a2, a3 ,a4
    model.cuda()
    model.eval()

    # # read muse dict
    # with open(dict_path) as f:
    #     for line in f:
    #         sw, tw = line.rstrip().split(' ', 1)
    #         if sw not in s2t_dict:
    #             s2t_dict[sw] = [tw]
    #         else:
    #             s2t_dict[sw].append(tw)

    # read dict
    s2t_dict_all = json.load(open(dict_path))
    # For each entry in dict, pick the top k translations
    s2t_dict = dict()
    for sw in s2t_dict_all:
        s2t_dict[sw] =  sorted(list(s2t_dict_all[sw].items()), key=lambda x: x[1], reverse=True)[:2]
        s2t_dict[sw] = [x[0] for x in s2t_dict[sw]]


    pre_1 = 0
    pre_2 = 0
    pre_5 = 0
    pre_10 = 0
    src_words = list(s2t_dict.keys())
    oov = 0
    
    s_l = []
    s_w = []
    c_l = []
    t_l = []
    for w in src_words:

        try:
            t_l = [s_tgt_w2i[wt] for wt in s2t_dict[w]]
            c_l.append(c_src_w2i[w])
            s_l.append(s_src_w2i[w])
            s_w.append(w)
        except:
            oov += 1
            pass
    # pdb.set_trace()
    static_src_id = torch.LongTensor(s_l)
    context_src_id = torch.LongTensor(c_l)
    '''
    static_src_id = torch.LongTensor([s_src_w2i[w] for w in src_words])
    context_src_id = torch.LongTensor([c_src_w2i[w] for w in src_words])
    '''
    if params.mode == 'v1':
        tgt_ids = model.test_all_word(static_src_id.cuda(), context_src_id.cuda(), None)
    else:
        vc_src_w2i = model.vecmap_context_src_dico.word2id
        vecmap_context_src_id = torch.LongTensor([vc_src_w2i[w] for w in s_w])
        tgt_ids = model.test_all_wordV2(static_src_id.cuda(), context_src_id.cuda(), vecmap_context_src_id.cuda(), params.lambda_w1)

    # Build output dictionary
    output_dict = dict()

    for i in range(len(tgt_ids)):
        output_dict[s_w[i]] = {model.static_tgt_dico[tgt_ids[i][j].item()]:(2-j) for j in range(10)}
        for j in range(10):
            if model.static_tgt_dico[tgt_ids[i][j].item()] in s2t_dict[s_w[i]]:
                if j == 0:
                    pre_1 += 1 
                    pre_5 += 1
                    pre_10 += 1
                    break
                
                elif j < 2:
                    pre_2 += 1
                    pre_5 += 1
                    pre_10 += 1
                    break

                elif j < 5:
                    pre_5 += 1
                    pre_10 += 1
                    break
                elif j < 10:
                    pre_10 += 1
                    break
    
    print('precision1:%f'%(pre_1/len(s_w)))
    print('precision2:%f'%(pre_2/len(s_w)))
    print('precision5:%f'%(pre_5/len(s_w)))
    print('precision10:%f'%(pre_10/len(s_w)))
    print('coverage:%f'%(len(s_w)/(len(s_w)+oov)))


    # Write output dictionary
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)


    # model = torch.load(model_path)
    # model.vecmap_context_src_dico, model.vecmap_context_src_embed, model.vecmap_context_tgt_dico, model.vecmap_context_tgt_embed = a1, a2, a3 ,a4
    # model.cuda()
    # model.eval()


test(params.model_path, params.dict_path, params.output_path)
        