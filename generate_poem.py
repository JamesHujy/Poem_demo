from jiuge_lvshi import Poem
from tqdm import tqdm

keywords = []
with open("keywords.txt",'r',encoding='utf-8') as f:
    keywords = f.readlines()
    keywords = [item.strip() for item in keywords]


qijue = []
wujue = []
poem = Poem()

for item in tqdm(keywords):
    wuyan = poem.generate(title=item, genre=0)
    qiyan = poem.generate(title=item, genre=1)
    qijue.append(item+' # '+ qiyan)
    wujue.append(item+' # '+ wuyan)

qijue = '\n'.join(qijue)
wujue = '\n'.join(wujue)


with open("qijue.txt", 'w', encoding='utf-8') as f:
    f.write(qijue)

with open("wujue.txt", 'w', encoding='utf-8') as f:
    f.write(wujue)