The labelled dataset is made of hand-labelled stock-related tweets downloaded from https://www.kaggle.com/utkarshxy/stock-markettweets-lexicon-data
Sentiment prediction was done using flair and SVMs. Several tweets with 'nan' label (3700) were discarded. Flair is a pre-trained
model and only makes positive and negative judgements, so neutral tweets were removed for that case. Flair had a 60% precision with
this stocks dataset. SVM had a f1 score of 58, 50, 42 for positive, neutral, and negative sentiments. 
