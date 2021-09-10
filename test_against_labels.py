import pandas as pd
import flair
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

training_set = pd.read_csv('tweets_dataset/tweets_labelled_09042020_16072020.csv',sep=';')
len_trn = training_set.shape[0]
drops = []
print(len_trn)
for i in range(0,len_trn):
	sent = training_set.iat[i,3]
	if(sent == 'neutral'):
		drops.append(i)
		continue
	if(sent != 'positive' and sent!='negative'):
		drops.append(i)

training_set = training_set.drop(labels=drops,axis=0)

# training_set.iat[i,0] is ID, training_set.iat[i,1] is created_at, .iat[i,2] is text of the tweet, iat[i,3] is the sentiment (positive or negative)
len_trn = training_set.shape[0]
print(len_trn)

#cleaning data
whitespace = re.compile(r"\s+")
web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
#tesla = re.compile(r"(?i)@Tesla(?=\b)")
user = re.compile(r"(?i)@[a-z0-9_]+")
pic_pattern = re.compile('pic\.twitter\.com/.{10}')
special_code = re.compile(r'(&amp;|&gt;|&lt;)')
tag_pattern = re.compile(r'<.*?>')
emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
STOPWORDS = set(stopwords.words('english')).union(
    {'rt', 'retweet', 'RT', 'Retweet', 'RETWEET'})
lemmatizer = WordNetLemmatizer()

def tokenize_stem(phrase):   
    tokens = word_tokenize(phrase)
    stem_words =[]
    for token in tokens:
        word = lemmatizer.lemmatize(token)
        stem_words.append(word)        
    buf = ' '.join(stem_words)    
    return buf

def remove_stopwords(phrase):
    return " ".join([word for word in str(phrase).split()\
                     if word not in STOPWORDS])

#clean_tweets = pd.DataFrame()
# we then use the sub method to replace anything matching
sentiment_model = flair.models.TextClassifier.load('en-sentiment')
probs = []
sentiments = []
comp = np.empty([len_trn,2], dtype=int)
agrees = np.zeros([len_trn,1], dtype=int)
for i in range(0,len_trn):
	tweet = training_set.iat[i,2]
#	print(tweet)
	tweet = whitespace.sub(' ', tweet)
	tweet = web_address.sub('', tweet)
	#tweet = tesla.sub('Tesla', tweet)
	tweet = user.sub('', tweet)
	tweet = pic_pattern.sub('', tweet)
	tweet = special_code.sub('',tweet)
	tweet = tag_pattern.sub('',tweet)
	tweet = emoji_pattern.sub('',tweet)
	tweet = tokenize_stem(tweet)
	tweet = remove_stopwords(tweet)
	training_set.iat[i,1] = tweet
#	print(tweet)
	orig_sent = training_set.iat[i,3]
	sentence = flair.data.Sentence(tweet)
	sentiment_model.predict(sentence)
	probability = sentence.labels[0].score  # numerical value 0-1
	sentiment = sentence.labels[0].value  # 'POSITIVE' or 'NEGATIVE'
	if(sentiment == 'POSITIVE'):
		comp[i,1] = 1
	elif(sentiment == 'NEGATIVE'):
		comp[i,1] = 0
	probs.append(probability)
	sentiments.append(sentiment)
	id = training_set.iat[i,0]
	if(orig_sent == 'positive'):
		comp[i,0] = 1
	elif(orig_sent == 'negative'):
		comp[i,0] = 0
	if(comp[i,0]==comp[i,1]):
		agrees[i] = 1

accuracy = (np.count_nonzero(agrees == 1))/len_trn
print(f"accuracy {accuracy}")

#	print(f"For tweet {id} the sentiment is {sentiment} with probability of {probability} vs originally {orig_sent}")
