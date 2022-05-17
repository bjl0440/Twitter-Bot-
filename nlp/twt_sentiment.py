import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy
from scipy.special import softmax

from twitter_api import tss

# roBERTa Model tokenizer and config
BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
config = AutoConfig.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

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

# Returns Score in percentage
def format_data(datapoint):
    result = []
    result.append('Negative: ' + str(round(datapoint[0]*100, 1)) + '%')
    result.append('Neutral: ' + str(round(datapoint[1]*100, 1)) + '%')
    result.append('Positive: ' + str(round(datapoint[2]*100, 1)) + '%')
    return result

# Returns result for data
def result(data):
    if data[0] > 0.60:
        return 'Negative'
    elif data[1] > 0.60:
        return 'Neutral'
    elif data[2] > 0.60:    
        return 'Positive'
    else:
        return 'Not Conclusive'

# Reads and formats Tweets csv file into dataframe
df = open("csv/tweets.csv", "r")
tweets_df = pd.read_csv('csv/tweets.csv',index_col=0)
tweets_df = tweets_df.reset_index()

x = 0
for tweet in tweets_df.loc[:,'Tweet']:
    x+=1
    input = tokenizer(format_tweet(tweet), return_tensors='pt')
    output = model(**input)
    sm = output[0][0].detach().numpy()
    score = softmax(sm).tolist()

    #testing
    print(x)
    print(format_data(score))
    print(result(score))
print(tss())