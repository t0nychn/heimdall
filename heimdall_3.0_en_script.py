# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:01:40 2021

@author: ztche
"""

# imports
import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import jieba
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import hanziconv
from hanziconv import HanziConv
import matplotlib
from matplotlib import pyplot

# seed
seed = 1
np.random.seed = seed

# load dataset
training_df = pd.read_excel('Weibo_data_set1.xlsx')
classes = training_df['N/O']
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# convert traditional Chinese characters to simplified
posts = training_df['Content']
posts1 = pd.DataFrame([HanziConv.toSimplified(p) for p in posts])

# prepare data for tokenization
def prep(str):
    """remove unwanted punctuation and stop words/noise from dataset, generalize cities and dates"""              
    cities = pd.read_csv('cities.csv')
    for city in cities["Chinese"]:
        if city in str:
            str = str.replace(city, " city ")                
    for character in ['月', '日']:
        if character in str:
            str = str.replace(character, " date ")  
    for word in ['我是', '我叫', '举报', '强制加班', '求助', '帮助', '爆料', 
                 '性骚扰', '拖欠', '超时', '强制', '投诉', '跳楼', '维权', '血汗']:
        if word in str:
            str = str.replace(word, " important ") 
    remove = ['[', ']', '【', '】', '#', '《', '》', '！', '!','，', '•',
              '/', '@', '。', '(', ')', '（', '）', ' ', '.', '%', '；',
              '：', '、', ':', '.', ',', '“', '”', '「', '」', ';','～',
              '‘', '?', '？', '…', '×', ' ', '-', '*', "'", '"', '’', '·',
              '\u200b', '的微博视频', '网页链接', '展开全文c', '分享自', '哈',
              '的', '和', '在', '了', '一', '为', '\ue627', 'O', '2', '996',
              '有', '中', '等', '上', '与', '对', '从', '不', '将', '到', '说',
              '地', '使', '年', '目前', '是', '百分之', '也', '还', '向',
              '并', '多','进行', '这些', '之后', '同', '一个', '这个', '下', 
              '而', '但是', 'H1B', '都', '第', '就', '个', '们', '富士康', '但', 
              '加班费', 'by','印度', 'Pro','我','4', '5', '9', '13', '6', '8', 
              '厂', '过', 'L', '10', '1', '3', '12', '20', '26', '40', '50',
              '分', '所', '0', '7', '苹果', '两', 'iPhone', '太', '该', '最', 
              'B', 'Gucci', 'inkedIn', '工', '农民', '这', '谁', '因', '为', 
              '没', '春节', '连', '完', '二', '超', '由', '加班', '只', '黑',
              '乡政府', '已', '成', '像', '昌硕', '如何', '派遣', '老板', '又']
    for r in remove:
        if r in str:
            str = str.replace(r, "")  
    return str
posts2 = pd.DataFrame([prep(p) for p in posts1[0]])

# tokenize
def space(str):
    """returns jieba.cut as a single string with whitespace"""
    str = " ".join([e for e in jieba.cut(str, cut_all=False)])
    return str
posts3 = pd.DataFrame([space(p) for p in posts2[0]])
all_words = []
for p in posts3[0]:
    words = word_tokenize(p)
    for w in words:
        all_words.append(w)
all_words = nltk.FreqDist(all_words)
most_common = [key for key, number in all_words.most_common(5000)]
word_features = list(most_common)

# build training feature-sets
def find_features(str):
    """find features function of each post using word_features"""
    words = word_tokenize(str)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features
posts4 = list(zip(posts3[0], Y))
np.random.shuffle(posts4)
featuresets = [(find_features(text), label) for text, label in posts4]

# build and train ensemble model
names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SGD Classifier', 'Naive Bayes', 'SVM Linear']
classifiers = [DecisionTreeClassifier(), RandomForestClassifier(), 
               LogisticRegression(), SGDClassifier(max_iter=100), MultinomialNB(), SVC(kernel='linear')]
models = list(zip(names, classifiers))
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, n_jobs=-1))
nltk_ensemble.train(featuresets)

# other functions
def clean(list):
    """returns .getall() list as uninterrupted text"""
    result= ""
    remove = ["\n"]
    for e in list:
        for r in remove:
            if r in e:
                e = e.replace(r, "")
        result += str(e)                
    yield result.strip()
    
def highlights(s):
    """use NLP model to highlight relevant posts"""
    predictions = []
    for e in s:
        e = HanziConv.toSimplified(e)
        e = prep(e)
        e = space(e)
        features = find_features(e)
        predictions.append(nltk_ensemble.classify(features))
    predictions = pd.Series(predictions) 
    is_useful = predictions == 0
    return ['background-color: #ffe5cc' if v else '' for v in is_useful]

# body of spider
class MySpider(scrapy.Spider):
    name = "Heimdall"
    
    def start_requests(self):
        
        # search term input
        terms = [term for term in input("Search Terms (space-seperated): ").split()] 
        for term in terms:
            urls = ['https://s.weibo.com/weibo?q=' + term + '&Refer=SWeibo_box']
            for url in urls:
                yield scrapy.Request(url = url, callback = self.parse)
                
    def parse(self, response):

        # selectors
        posts = response.css( '.content' )
        author = posts.css( '.name' )
        text = posts.xpath( './/p[@node-type="feed_list_content"]' )
        info = posts.css( '.from' )
        search_term = response.css( 'div.search-input ::attr(value)' ).get()

        # counter
        x = -1

        # display search info
        print("\nSearch Term: " + str(search_term))
        print("Results: " + str(len(text)))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Search Completed: " + str(date.today()) + " at " + str(current_time))

        # select search contents
        author_data = []
        text_data = []
        info_data = []
        for t in text:
            x += 1
            counter_author = author[x]
            counter_content = text[x] 
            counter_info = info[x]
            author_extracts = counter_author.css( '::text' ).getall()
            text_extracts = counter_content.css( '::text' ).getall()
            info_extracts = counter_info.css( '::text' ).getall()
            
            # append data
            author_data.append(''.join(str(e).strip() for e in author_extracts))
            text_data.append(''.join(str(e).strip() for e in text_extracts))
            info_data.append(''.join(str(e).strip() for e in info_extracts))
        
        # visualise using DataFrame
        data = {'Author' : author_data,
                'Content' : text_data,
                'Info' : info_data
                }
        df = pd.DataFrame(data, columns = ['Author', 'Content', 'Info'])
        df.index = np.arange(1, len(df)+1)
        pd.set_option('display.max_colwidth', None)
        display(df.style.apply(highlights, subset=["Content"]))
        
# run spider
process = CrawlerProcess()
process.crawl(MySpider)
process.start()