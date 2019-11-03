import torch
import pytorch_pretrained_bert
import tokenization_bert
import json

from pytorch_pretrained_bert import GPT2LMHeadModel
from generator import CheckedGenerator
from poemcheck import Checker


class Poem():
    def __init__(self, model_path=None, model_config=None):
        self.model = None
        if(model_path is not None and model_config is not None):
            self.load_model(model_path, model_config)
        else:
            self.load_model()
        self.tokenizer = tokenization_bert.BertTokenizer(
            vocab_file='cache/vocab_with_title.txt')
        self.checker = Checker()
        with open('./cache/label_to_id.json', 'r', encoding='utf-8') as f:
            self.title_to_ids = json.load(f)

    def load_model(self, model_path='./cache/model/model_epoch_1.pt',
                   model_config='./cache/model_config.json', device='cpu'):
        # /data/disk1/private/hujinyi/gpt_poem/model_with_title/model_epoch_1.pt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_config = pytorch_pretrained_bert.GPT2Config.from_json_file(
            model_config)
        model_state_dict = torch.load(model_path)
        model = GPT2LMHeadModel(config=model_config)
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def generate(self, title='无题', genre=2):
        if(self.model is None):
            raise Exception("has no model")

        temperature = 1
        topk = 15

        context_tokens = []
        assert genre in [0,1,2,3]

        text_genre_list = ['五言绝句','七言绝句','五言律诗','七言律诗']
        genre_code_list = ['wuyanjue', 'qiyanjue', 'wuyanlv', 'qiyanlv']

        text_genre = text_genre_list[genre]
        genre_code = genre_code_list[genre]

        ids = self.title_to_ids[text_genre]
        context_tokens.append(ids)

        context_tokens.append(100)
        context_tokens.extend(
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(title)))
        context_tokens.append(4282)  # 4282 is #

        out = None
        while out is None:
            generator = CheckedGenerator(model=self.model,
                                         context=context_tokens,
                                         tokenizer=self.tokenizer,
                                         checker=self.checker,
                                         genre=genre_code,
                                         temperature=temperature,
                                         top_k=topk, device=self.device)
            out = generator.sample_sequence()
        out = out.tolist()

        text = self.tokenizer.convert_ids_to_tokens(out[0])
        text = text[:-1]
        text = ''.join(text)
        text = text.split('#')[-1]
        return text


if __name__ == '__main__':
    lvshi = Poem()
    print(lvshi.generate())
