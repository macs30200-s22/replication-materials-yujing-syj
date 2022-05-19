from build_user_subreddit_history import read_json_list
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from community import community_louvain
import matplotlib.cm as cm
from datetime import datetime
from datetime import date
import datetime as dt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import spacy
spacy.prefer_gpu()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
import statsmodels.formula.api as smf
from gensim.models import TfidfModel
import json
import glob
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models as gensim_models
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Select variables that I want related to the posts: time, text, date
def get_contents(sub):
    
    result_dic = {}
    filename = "new_data/{1}/{0}/{1}_jsonlists.gz".format(sub, 'posts')
    i = 0
    for dic in read_json_list(filename):
            result = {}
            if 'selftext' in dic:
                title = dic['title']
                text = dic['selftext']
                subreddit = dic['subreddit']
                created_time = dic['created_utc']   
                if text != '[removed]'and text !='[deleted]':
                    result['text'] = title+' '+text 
                    result['time'] = datetime.utcfromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
                    result['date'] = result['time'][:10]
                    result_dic[i] = result
                    i += 1
    return result_dic

workfromhome = get_contents('WorkFromHome')

def sentiment_scores(sentence):
    
    result = 0
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.35 :
        result = "Positive"
 
    elif sentiment_dict['compound'] <= - 0.05 :
        result = "Negative"
 
    else :
        result = "Neutral"

    return result

for k,v in workfromhome.items():
    workfromhome[k]["sentiment"] = sentiment_scores(v['text'])


year_month = ['2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04']

dic = {}
for i in year_month:
    dic[i] = {}
    dic[i]['positive'] = 0
    dic[i]['negative'] = 0
    dic[i]['neutral'] = 0
    dic[i]['num'] = 0  
    for k,v in workfromhome.items():
        if v['date'][:7] == i:
            dic[i]["num"] += 1
            result = workfromhome[k]["sentiment"]
            if result == "Positive":
                dic[i]['positive'] += 1
            elif result == "Negative":
                dic[i]['negative'] += 1
            else:
                 dic[i]['neutral'] += 1
    

for k,v in dic.items():
    dic[k]["positive_rate"] = v['positive']/v['num']
    dic[k]["negative_rate"] = v['negative']/v['num']
    dic[k]["neutral_rate"] = v['neutral']/v['num']


values_num = [list(dic.values())[i]['num'] for i in range(len(dic.keys()))]
values_positiverate = [list(dic.values())[i]['positive_rate'] for i in range(len(dic.keys()))]
values_negativerate= [list(dic.values())[i]['negative_rate'] for i in range(len(dic.keys()))]
values_neutralrate = [list(dic.values())[i]['neutral_rate'] for i in range(len(dic.keys()))]
df = pd.DataFrame(year_month, columns = ['Time'])
df['Posts(num)'] = values_num
df['posts positive rate'] = values_positiverate
df['posts negative rate'] = values_negativerate
df['posts neutral rate'] = values_neutralrate

covid_cases = pd.read_csv('Covid-19 new cases.csv')  
df['US covid-19 New Cases'] = covid_cases['US New Cases']
df['Global covid-19 New Cases'] = covid_cases['Global New Cases']

df.to_csv('reddits sentiment and covid-19.csv')



