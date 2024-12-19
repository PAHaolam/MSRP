import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from LanguageModel import LanguageModel
from SemanticSimilarity import SemanticSimilarity

def generate_summary(model, text, tl, tokenizer, lm_model, ss_model, num_beams, max_length, min_length):
    
    stopwords = ['in', 'at', 'to', 'on', 'the', "'s", 'of', 'a', 'for', 'with', 'is', 'into', 'by', 'his', 'her', 'when', 'and', 'but']
        
    dayofweek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tids = tokenizer(text.strip(), add_special_tokens=True).input_ids
    tids = torch.LongTensor([tids])
    tids = tids.to(device)

    MAXLEN = tl * max_length                 
    MINLEN = int(tl * min_length)                           

    attmask = (tids != tokenizer.pad_token_id)
    bo = model.generate(input_ids=tids, do_sample=False, min_length=MINLEN,
                        max_length=MAXLEN, attention_mask=attmask,
                        no_repeat_ngram_size=3, num_beams=num_beams,
                        num_return_sequences=num_beams,
                        early_stopping=False)
    
    str_bo = tokenizer.batch_decode(bo, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False)
    
    for idx, s in enumerate(str_bo):   
        for _ in range(5):
            if len(s.split()) > 1 and s.split()[-1] in stopwords: 
                s = ' '.join(s.split()[:-1])

        for dw in dayofweek:
            dw = dw.lower()
            if dw in s: 
                s = s.replace(dw, '') 

        str_bo[idx] = s
    
    # Content preservation & Fluency
    s_score = ss_model.get_ss_score(str_bo, [text]*num_beams).cpu().numpy()
    l_score = lm_model.get_lm_score(str_bo).cpu().numpy()                      

    # Length 
    tlens = np.array([len(t.split()) for t in str_bo])                        
    lenerr= abs(tlens-tl)                                         
    length_penalty = -(lenerr)  

    final_score = s_score + l_score + length_penalty
    
    best_s = str_bo[final_score.argmax()]
    
    return best_s

fn = sys.argv[1]
gpu = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"]=gpu

tokenizer = AutoTokenizer.from_pretrained('t5-small') 
model = T5ForConditionalGeneration.from_pretrained(fn).cuda()

if fn.endswith('_sb'):
    ssmt = 'sbert'
else:
    ssmt = 'sent2vec'

ss_model = SemanticSimilarity(model_type=ssmt)
lm_model = LanguageModel()    

max_length = 3
min_length = 1.5 if 'ratio' not in fn else 1.9
num_beams = 20

if __name__ == "__main__":
    q = int(input())
    for _ in range(q):

        tl = int(input())
        if(tl <= 0):
            print("target length should be greater than 0.")
            continue

        text = input()
        if(tl >= len(text.split())):
            print("target length should be greater than text length.")
            continue

        pred = generate_summary(model, text, tl, tokenizer, lm_model, ss_model, num_beams, max_length, min_length)

        print(pred)
        print(len(pred))