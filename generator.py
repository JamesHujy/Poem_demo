import torch
import torch.nn.functional as F
import time
import numpy as np
class BaseGenerator(object):
    def __init__(self, model, context, tokenizer, temperature=1, top_k=0, device='cpu'):
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        self.generated = self.get_context_tensor(context)

    def filtering(self, logits, filter_value=-float('Inf')):
        assert logits.dim() == 1  
        if self.top_k > 0:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
            top_values, top_indices = torch.topk(logits, 30, dim=-1)
            generated = self.generated[0].tolist()
            generated = [index for index in generated if index != 0 and index != 3946]
            length = len(generated)
            #if generated[-1] != generated[-2]:
                #generated = generated[:-1]
            for index in generated:
                logits[index] = filter_value
        return logits

    def sample_sequence(self):
        with torch.no_grad():
            while True:
                inputs = {'input_ids': self.generated}
                outputs = self.model(**inputs)  
                next_token_logits = outputs[0][0, -1, :] / self.temperature
                filtered_logits = self.filtering(next_token_logits)
                while True:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token.tolist() != [6809]:
                        break
                if next_token.tolist() == [0]:
                    break
                self.generated = torch.cat((self.generated, next_token.unsqueeze(0)), dim=1)
        return self.generated

    def get_context_tensor(self, context):
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)
        return context


class CheckedGenerator(BaseGenerator):
    def __init__(self, model, context, tokenizer, checker, genre, temperature=1, top_k=0, device='cpu'):
        super(CheckedGenerator, self).__init__(model, context, tokenizer, temperature, top_k, device)
        self.checker = checker
        self.pattern_label = None
        self.pattern = None
        self.genre = genre
        self.subgenre = 'lv' if len(genre) == 7 else 'jue'
        self.genre_to_length = {"wuyanlv": 5, "qiyanlv": 7, "wuyanjue": 5, "qiyanjue": 7}
        self.count = -1
        self.position = 0
        self.yun = None

    def filtering_with_check(self, logits, filter_value=-float('Inf')):
        assert logits.dim() == 1
        interval_length = self.genre_to_length[self.genre]
        tokens = (self.tokenizer).convert_ids_to_tokens((self.generated)[0].tolist())
        if tokens[-1] == '，':
            self.position = 0
        else:
            self.position += 1
        if self.pattern_label == None and tokens[-1] == '，':
            self.pattern_label = self.checker.judge_pattern(tokens[(-1-interval_length):-1], self.subgenre)
            if self.pattern_label == None:
                return None
            pingze = self.pattern_label.split(' ')[0][-1]
            if pingze == '1':
                self.yun = tokens[-2]
            self.pattern = self.checker.getpattern(self.pattern_label, self.subgenre)
            return self.filtering_with_labels(logits)
        else:
            if tokens[-1] == '，' and self.yun == None:
                self.yun = tokens[-2]
            
            if self.pattern_label == None:
                return super().filtering(logits)
            else:
                return self.filtering_with_labels(logits)

    def filtering_with_labels(self, logits):
        self.count += 1
        interval_length = self.genre_to_length[self.genre]
        if self.count >= len(self.pattern):
            return super().filtering(logits)
        else:
            current = self.pattern[self.count]
            if current == ' ' or current == '0':
                return super().filtering(logits)
            else:
                if current == '1':
                    if self.position == interval_length - 1 and self.yun != None:
                        logits = self.filteringYun(logits)
                    return self.filteringPing(logits)
                else:
                    return self.filteringZe(logits)

    def filteringPing(self, logits, filter_value=-float('Inf')):
        pingindex = self.checker.get_zeindex(self.tokenizer)
        logits[pingindex] = filter_value
        return super().filtering(logits)

    def filteringZe(self, logits, filter_value=-float('Inf')):
        zeindex = self.checker.get_pingindex(self.tokenizer)
        logits[zeindex] = filter_value
        return super().filtering(logits)

    def filteringYun(self, logits, filter_value=-float('Inf')):
        entire = set(range(len(logits)))
        yun = self.checker.get_yunindnex(self.yun, self.tokenizer)
        left = entire - set(yun)
        logits[list(left)] = filter_value
        return logits

    def sample_sequence(self):
        with torch.no_grad():
            while True:
                inputs = {'input_ids': self.generated}
                outputs = self.model(**inputs)  
                next_token_logits = outputs[0][0, -1, :] / self.temperature
                filtered_logits = self.filtering_with_check(next_token_logits)
                if filtered_logits is None:
                    return None
                while True:
                    logits = np.array(F.softmax(filtered_logits, dim=-1).tolist())
                    logits = logits[logits > 0]
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token.tolist() != [6809]:
                        break
                if next_token.tolist() == [0]:
                    break
                self.generated = torch.cat((self.generated, next_token.unsqueeze(0)), dim=1)
        return self.generated







































