import json

class Checker(object):
    def __init__(self):

        self.five_lv_sentence_pattern = ["01022", "11021", "02012", "02211"]
        self.seven_lv_sentence_pattern = ["0102012", "0102211", "0201022", "0211021"]

        self.five_jue_sentence_pattern = ["01122", "11221", "02112", "02211"]
        self.seven_jue_sentence_pattern = ["0102112", "0102211","0201122","0211221"]

        self.seven_lv_pattern = {"0102012": "0211021 0201022 0102200 0102012 0211021 0201022 0102211",
                              "0102211": "0211021 0201022 0102211 0102012 0211021 0201022 0102211",
                              "0201022": "0102211 0102012 0211021 0201022 0102211 0102012 0211021",
                              "0211021": "0102211 0102012 0211021 0201022 0102011 0102012 0211021"}  
        self.five_lv_pattern = {"01022": "02211 02012 11021 01022 02211 02012 11021",
                             "11021": "02211 02012 11021 01022 02211 02012 11021",
                             "02012": "11021 01022 02211 02012 11021 01022 02211",
                             "02211": "11021 01022 02211 02012 11021 01022 02211"}

        self.five_jue_pattern = {"01122": "02211 02112 11221",
                                 "11221": "02211 02012 11221",
                                 "02112": "11221 01122 02211",
                                 "02211": "11221 01122 02211"}
                                 
        self.seven_jue_pattern = {"0102112": "0211221 0201122 0102211",
                                 "0102211": "0211221 0201122 0102211",
                                 "0201122": "0102211 0102112 0211221",
                                 "0211221": "0202211 0102112 0211221"}

        self.yundict = self.initial_yundic()
        with open("./cache/zetable.txt", "r", encoding="utf-8") as f:
            self.zetable = f.read()
        with open("./cache/pingtable.txt", "r", encoding="utf-8") as f:
            self.pingtable = f.read()

    def judge_pattern(self, sentence, genre):
        ans = ""
        for item in sentence:
            if item in self.pingtable:
                ans += "1"
            else:
                ans += "2"
        if genre == "lv":
            pattern_collection = self.five_lv_sentence_pattern if len(sentence) == 5 else self.seven_lv_sentence_pattern
        else:
            pattern_collection = self.five_jue_sentence_pattern if len(sentence) == 5 else self.seven_jue_sentence_pattern
        for pattern in pattern_collection:
            if self.match(pattern, ans):
                return pattern
        return None

    def match(self, pattern, item):
        for i in range(len(item)):
            if pattern[i] == '0':
                continue
            elif pattern[i] != item[i]:
                return False
        return True

    def get_zeindex(self, tokenizer):
        return tokenizer.convert_tokens_to_ids(list(self.zetable))

    def get_pingindex(self, tokenizer):
        return tokenizer.convert_tokens_to_ids(list(self.pingtable))

    def getpattern(self, label, genre):
        if label == None:
            return None
        else:
            if genre == "lv":
                if len(label) == 5:
                    return self.five_lv_pattern[label]
                else:
                    return self.seven_lv_pattern[label]
            else:
                if len(label) == 5:
                    return self.five_jue_pattern[label]
                else:
                    return self.seven_jue_pattern[label]

    def checktype(self, word):
        if word in self.pingtable:
            return 1
        elif word in self.zetable:
            return 2
        return None

    def initial_yundic(self):
        with open('./cache/yun.json','r',encoding='utf-8') as f:
            data = json.load(f)
            return data

    def get_yunindnex(self, word, tokenzier):
        candidate = list(self.yundict[word])
        return tokenzier.convert_tokens_to_ids(candidate)
 
       