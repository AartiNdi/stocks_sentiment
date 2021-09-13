import pandas as pd
#import flair
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

training_set = pd.read_csv('tweets_dataset/tweets_labelled_09042020_16072020.csv',sep=';')
len_trn = training_set.shape[0]
drops = []
print(len_trn)
for i in range(0,len_trn):
	sent = training_set.iat[i,3]
	if(sent != 'positive' and sent!='negative' and sent!='neutral'):
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
#sentiment_model = flair.models.TextClassifier.load('en-sentiment')
probs = []
sentiments = []
orig_sent = []
orig_sent_num = np.empty([len_trn,1], dtype=int)
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
	training_set.iat[i,2] = tweet
#	print(tweet)
	orig_sent = training_set.iat[i,3]
	if(orig_sent == 'positive'):
		orig_sent_num[i] = 1
	elif(orig_sent == 'negative'):
		orig_sent_num[i] = -1
	elif(orig_sent == 'neutral'):
		orig_sent_num[i] = 0

pred_set = pd.read_csv('tweets_dataset/tweets_remaining_09042020_16072020.csv',sep=';')
for i in range(0,len_trn):
	tweet = pred_set.iat[i,2]
#       print(tweet)
	tweet = whitespace.sub(' ', tweet)
	tweet = web_address.sub('', tweet)
# tweet = tesla.sub('Tesla', tweet)
	tweet = user.sub('', tweet)
	tweet = pic_pattern.sub('', tweet)
	tweet = special_code.sub('',tweet)
	tweet = tag_pattern.sub('',tweet)
	tweet = emoji_pattern.sub('',tweet)
	tweet = tokenize_stem(tweet)
	tweet = remove_stopwords(tweet)
	pred_set.iat[i,2] = tweet

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(training_set.iloc[:,2])
pred_vectors = vectorizer.transform(pred_set.iloc[:,2])
X_train, X_test, y_train, y_test = train_test_split(train_vectors, np.ravel(orig_sent_num), test_size=0.33, random_state=42)
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(X_train, y_train)
test_linear = classifier_linear.predict(X_test)
agrees = test_linear - y_test
precision = 1 - (np.count_nonzero(agrees)/agrees.shape[0])
print(f"precision {precision}")
report = classification_report(y_test, test_linear, output_dict=True)
print('positive: ', report['1'])
print('neutral: ', report['0'])
print('negative: ', report['-1'])
#prediction_linear = classifier_linear.predict(pred_vectors)
