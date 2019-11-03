from django.shortcuts import render
import requests
import sys
from subprocess import run,PIPE

from .jiuge_lvshi import Poem
from watchprob import get_prob
from watchprob_single import get_prob as get_prob_single

def button(request):
    genreList = ['七言律诗','七言绝句','五言律诗','五言绝句']
    with open('poem.txt','w',encoding='utf-8') as f:
        f.write('')
    return render(request,'home.html',{'genreList':genreList})

def external(request):
    poem = Poem()
    topic = request.POST.get('topic')
    genre = request.POST.get('genre')
    genre_dict = {'wuyanjue':0,'qiyanjue':1,'wuyanlv':2,'qiyanlv':3}
    genre = genre_dict[genre]
    generate = poem.generate(topic,genre=genre)
    with open('poem.txt','w',encoding='utf-8') as f:
        f.write(str(genre)+"#"+topic+"#"+generate)  
    generate = generate.strip().split('，')
    if genre == 0:
        Row = list(range(5))[1:]
        Column = list(range(6))[1:]
    if genre == 1:
        Row = list(range(5))[1:]
        Column = list(range(8))[1:]
    if genre == 2:
        Row = list(range(9))[1:]
        Column = list(range(6))[1:]
    if genre == 3:
        Row = list(range(9))[1:]
        Column = list(range(8))[1:]
    generate.insert(0, topic)
    return render(request,'home.html',{'topic':topic,'Poem':generate,'generated':True,'Row':Row,'Column':Column})

def getPoem(poem, row, column):
    poem = poem.split('，')
    history = '，'.join(poem[:row-1])
    history += '，'+poem[row-1][:column]
    return history

def getPoem_single(poem, row, column):
    poem = poem.split('，')
    history = poem[row-1][:column]
    return history

def checkprob(request):
    with open('poem.txt', 'r' ,encoding='utf-8') as f:
        content = f.read().strip().split("#")

    genreList = ['七言律诗','七言绝句','五言律诗','五言绝句']
    genre = int(content[0])
    if genre == 0:
        Row = list(range(5))[1:]
        Column = list(range(6))[1:]
    if genre == 1:
        Row = list(range(5))[1:]
        Column = list(range(8))[1:]
    if genre == 2:
        Row = list(range(9))[1:]
        Column = list(range(6))[1:]
    if genre == 3:
        Row = list(range(9))[1:]
        Column = list(range(8))[1:]

    genre = genreList[genre]
    topic = content[1]
    poem = content[-1]

    number = int(request.POST.get('prob_num'))
    row = int(request.POST.get('Rowselected'))
    column = int(request.POST.get('Columnselected'))
    history = getPoem(poem, row, column)
    history_single = getPoem_single(poem, row, column)
    print(history)
    print(history_single)
    distrib = get_prob(history, number, genre, topic)
    distrib_single = get_prob_single(history, number, genre, topic)
    print(distrib)

    poem = poem.split('，')
    poem.insert(0, topic)
    return render(request,'home.html',{'topic':topic,'Poem':poem,'generated':True,'Row':Row,'Column':Column,
        'PoemModel':distrib,'LangModel':distrib_single,'poem_model_history':history,'lang_model_history':history_single})



