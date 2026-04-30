from face_score import FACEScorer
from data_utils import load_text_data
from tqdm import tqdm
import torch
import pandas as pd
import re
from tqdm import tqdm

def test_pythia_nll(text_file,save_dir="/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-a-nll.txt"):
    model_path = '/data1/model/llama-7b-base'
    # model_path = '/data1/model/pythia-410m-base'
    scorer = FACEScorer(model_path, device='cuda:0')
    texts = load_text_data(text_file)
    pack = scorer.texts_to_nll_with_tokens(texts)
    
    # process tokens and words
    # if further one word nll is needed
    nll_all = []
    for i in tqdm(range(len(pack))):
        words = texts[i].split()
        tokens = pack[i]['tokens']
        nlls = pack[i]['nll']
        nll_list = []
        # remove special characters and leading space
        tokens = [re.sub(r'[^\w\s]', '', token.lstrip('Ġ')) for token in tokens]
        # remove the last token if it is a 'Ċ'
        tokens = tokens[:-1] if tokens[-1] == 'Ċ' else tokens
        words = [re.sub(r'[^\w\s]', '', word) for word in words]
        # remove empty string
        tokens = [token for token in tokens if token.strip()]
        words = [word for word in words if word.strip()]
        
        # remove the first word
        words = words[1:]
        token_index = 0
        for i in range(len(words)):
            temp = tokens[token_index]
            nll_plus = nlls[token_index]
            while temp not in words[i]:
                print("temp not in words[i]",temp,words[i])
                token_index += 1
                temp = tokens[token_index]
            while temp != words[i]:
                temp += tokens[token_index+1]
                token_index += 1
                nll_plus += nlls[token_index]
            nll_list.append((words[i],nll_plus))
            # nll_list.append({'word':words[i],'nll':nll_plus,})
            token_index += 1
        nll_all.append(nll_list)  
    # pack = [{'nll':nll,'tokens':tokens} for nll,tokens in pack]
    # pd.DataFrame(pack).to_csv(save_dir,header=None,index=None)
    
    # sum of nll
    nlls_sum = [sum(nll['nll']) for nll in pack]
    pd.DataFrame(nlls_sum).to_csv(save_dir,header=None,index=None)
    return nll_all

nlls_a = test_pythia_nll("/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-a.txt","/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-a-nll.txt")
nlls_b = test_pythia_nll("/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-b.txt","/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-b-nll.txt")
pd.DataFrame(nlls_a).to_csv("/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-a-nlls-1.txt",header=None,index=None)
pd.DataFrame(nlls_b).to_csv("/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-b-nlls-1.txt",header=None,index=None)
# test_pythia_nll("/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-b.txt","/home/lxq/proj/eye-track/manipulation/metrics/mt_exp_dataset/nll/nll_FACEScore/mt-bench-b-nll.txt")