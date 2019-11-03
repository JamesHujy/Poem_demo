import torch
import torch.nn.functional as F
import argparse
import json
import tokenization_bert
import pytorch_pretrained_bert
import os

from pytorch_pretrained_bert import GPT2LMHeadModel

class WatchProb(object):
    def __init__(self, model, context, tokenizer, temperature=1, top_k=0, device='cpu'):
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        self.generated = self.get_context_tensor(context)


    def get_prob(self):
        with torch.no_grad():
            inputs = {'input_ids': self.generated}
            outputs = self.model(**inputs)  
            next_token_logits = outputs[0][0, -1, :] / self.temperature
            softmax_logits = F.softmax(next_token_logits)
        return softmax_logits

    def show_prob(self, topk=100):
        softmax_logits = self.get_prob()
        top_value, top_indices = torch.topk(softmax_logits, topk, dim=-1)

        top_value = top_value.tolist()
        top_indices = top_indices.tolist()

        text = self.tokenizer.convert_ids_to_tokens(top_indices)

        prob = {}
        result = []
        for i in range(topk):
            if text[i] != '[PAD]':
                prob[text[i]] = top_value[i]
        for i, (key, value) in enumerate(prob.items()):
            temp = str(i) + ":" + str(key) + ":" + str(round(value, 5))
            result.append(temp)
        return result

    def show_cumulative(self, cumu_num):
        softmax_logits = self.get_prob()
        softmax_logits, indices = torch.sort(softmax_logits, descending=True)
        softmax_logits = softmax_logits.tolist()
        accumu_sum = 0

        for i in range(len(softmax_logits)):
            accumu_sum += softmax_logits[i]
            if accumu_sum >= cumu_num:
                return i+1

    def get_context_tensor(self, context):
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)
        return context
                
def get_prob(context, topk, genre, title):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab_fine_tuning.txt')

    model_config = pytorch_pretrained_bert.GPT2Config.from_json_file('cache/model_config_single.json')
    model_state_dict = torch.load('cache/model_single/model_epoch_1.pt')

    model = GPT2LMHeadModel(config=model_config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    batch_size = 1
    temperature = 1

    context_tokens = []

    with open('./cache/label_to_id.json','r',encoding='utf-8') as f:
        title_to_ids = json.load(f)
    try:
        ids = title_to_ids[genre]
        context_tokens.append(ids)
    except:
        ids = title_to_ids['七言律诗']
        context_tokens.append(ids)

    context_tokens.append(100)
    context_tokens.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title)))
    context_tokens.append(4282) # 4282 is #

    raw_text = context
    if raw_text != "":
        context_tokens.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text)))


    watcher = WatchProb(model=model, context=context_tokens, tokenizer=tokenizer, temperature=temperature, top_k=topk, device=device)
    prob_dis = watcher.show_prob(topk=topk)

    eight_cumu = watcher.show_cumulative(0.8)
    nine_cumu = watcher.show_cumulative(0.9)
    ninefive_cumu = watcher.show_cumulative(0.95)
    prob_dis.append("")
    prob_dis.append("")
    prob_dis.append("0.8累计覆盖: "+str(eight_cumu))
    prob_dis.append("0.9累计覆盖: "+str(nine_cumu))
    prob_dis.append("0.95累计覆盖: "+str(ninefive_cumu))

    return prob_dis

if __name__ == '__main__':
    pass


                
        

    








































