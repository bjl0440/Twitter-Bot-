import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# formats tweet for analysis
def format_tweet(tweet):
    text = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = "@user"
        if word.startswith('http') and len(word) > 1:
            word = 'http'
        text.append(word)
    return ' '.join(text)

#df = open("csv/tweets.csv", "r")
tweets_df = pd.read_csv('csv/tweets.csv',index_col=0)
tweets_df = tweets_df.reset_index()

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

for tweet in tweets_df.loc[:,'Tweet']:
    print(format_tweet(tweet))