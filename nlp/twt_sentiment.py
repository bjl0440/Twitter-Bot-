import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification



#df = open("csv/tweets.csv", "r")
tweets_df = pd.read_csv('csv/tweets.csv',index_col=0)
tweets_df = tweets_df.reset_index()

tweet_number = 0

# print(tweets_df)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

for tweet in tweets_df.loc[:,'Tweet']:
    for word in tweet.split(' '):
        if word.startswith('@') and word.len() > 1:
            word = "@user"
        if word.startswith('https:'):
            word = 'https://'
        if word.startswith('#'):
            word = '#Hashtag'