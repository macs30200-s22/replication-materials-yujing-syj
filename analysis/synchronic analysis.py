### Import all the packages
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
import json
import glob
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
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

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    raw_texts = []
    for text in texts:
        raw_texts.append(text)
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        final = re.sub('https?:[^\s]+', '', final)
        texts_out.append(final)
    return texts_out,raw_texts

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

def make_bigrams(texts):
    return([bigram[doc] for doc in texts])

def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

# def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=raw_texts):
#     # Init output
#     sent_topics_df = pd.DataFrame()

#     # Get main topic in each document
#     for i, row in enumerate(ldamodel[corpus]):
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)
#         # Get the Dominant topic, Perc Contribution and Keywords for each document
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:  # => dominant topic
#                 wp = ldamodel.show_topic(topic_num)
#                 topic_keywords = ", ".join([word for word, prop in wp])
#                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
#             else:
#                 break
#     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

#     # Add original text to the end of the output
#     contents = pd.Series(texts)
#     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
#     return(sent_topics_df)

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

    return result,sentiment_dict['compound']

workfromhome = get_contents('WorkFromHome_3years')
workfromhome1 = []
for k,v in workfromhome.items():
    workfromhome1.append(v['text'])
lemmatized_texts, raw_texts = lemmatization(workfromhome1)
data_words = gen_words(lemmatized_texts)

#BIGRAMS AND TRIGRAMS
bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=30)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=30)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

stopwords = stopwords.words("english")

word_lst = ['get','just','go','so','do','take','make','give','vs','also','let','th','amp','etc','ve',"seem",       "think","look","thing","else","already","turn","way","see","say","one","come","bit","actually","set","put","sure","lot","even","maybe","whole","ensure","co"]

new_data_bigrams_trigrams = []
for lst in data_bigrams_trigrams:
    new_lst = []
    for i in lst:
        if i not in word_lst and i not in stopwords:
            new_lst.append(i)
    new_data_bigrams_trigrams.append(new_lst)

new_data_bigrams_trigrams = []
for lst in data_bigrams_trigrams:
    new_lst = []
    for i in lst:
        if i not in word_lst and i not in stopwords:
            if i == 'coronavirus' or i == 'pandemic':
                new_lst.append('covid')
            elif i == 'workfromhome':
                new_lst.append('wfh')
            elif i == 'remotely':
                new_lst.append('remote')
            elif i == 'stand_desk':
                new_lst.append('standing_desk')
            else:
                new_lst.append(i)
    new_data_bigrams_trigrams.append(new_lst)

data_bigrams_trigrams = new_data_bigrams_trigrams

#TF-IDF REMOVAL
id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.1
words  = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                               id2word=id2word,
                               num_topics=12,
                               random_state=100,
                               update_every=1,
                               chunksize=100,
                               passes=10,
                               alpha="auto")

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=raw_texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=raw_texts)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

topic_dic = {}
for row in range(len(df_dominant_topic)):
    if df_dominant_topic.iloc[row,2] >0.2:
        if df_dominant_topic.iloc[row,1] not in topic_dic.keys():
            topic_dic[df_dominant_topic.iloc[row,1]]=[]
        topic_dic[df_dominant_topic.iloc[row,1]].append(df_dominant_topic.iloc[row,4])



topic_sentiment_dic = {}
for k,v in topic_dic.items():
    topic_sentiment_dic[k]={}
    topic_sentiment_dic[k]["Positive"] = 0
    topic_sentiment_dic[k]["Neutral"] = 0
    topic_sentiment_dic[k]["Negative"] = 0
    topic_sentiment_dic[k]["Num"] = 0    
    for i in v:
        topic_sentiment_dic[k]["Num"] += 1 
        if sentiment_scores(i)[0] =="Positive":
            topic_sentiment_dic[k]["Positive"] +=1
        elif sentiment_scores(i)[0] =="Neutral":
            topic_sentiment_dic[k]["Neutral"] +=1        
        else:
            topic_sentiment_dic[k]["Negative"] +=1    
    topic_sentiment_dic[k]["Positive Rate"] = topic_sentiment_dic[k]["Positive"]/topic_sentiment_dic[k]["Num"]
    topic_sentiment_dic[k]["Neutral Rate"] = topic_sentiment_dic[k]["Neutral"]/topic_sentiment_dic[k]["Num"]
    topic_sentiment_dic[k]["Negative Rate"] = topic_sentiment_dic[k]["Negative"]/topic_sentiment_dic[k]["Num"]
    
topic_name = {6:"find wfh jobs, interviews & experience for jobs",
             5:"company's requirement & work experience",
             7:"technology equipments issues",
             3:"team, coworkers & boss",
             9:"make money online & ads",
             10:"equipment (desk, chair) & health",
             2:"life style & work-life balance",
             11:"back to office & physical and mental health",
             8:"surveys",
             0:"recommendation (lonely, productivity, etc.)",
             1:"time management",
             4:"home & family"}


topic_sentiment_df = pd.DataFrame(columns=['dominant topic','topic name','nums','positive rate','neutral rate','negative rate'])
for i in topic_dic.keys():
    topic_sentiment_df = topic_sentiment_df.append({'dominant topic': i,'topic name':topic_name[i],'nums':len(topic_dic[i]), 'positive rate':topic_sentiment_dic[i]["Positive Rate"] ,'neutral rate':topic_sentiment_dic[i]["Neutral Rate"],'negative rate':topic_sentiment_dic[i]["Negative Rate"]}, ignore_index=True)
    
topic_sentiment_df.to_csv('topic sentiment(wfh) raw1.csv')    
    
    
    
    
    
    
    
    
