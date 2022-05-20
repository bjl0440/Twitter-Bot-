import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy
from scipy.special import softmax
import matplotlib.pyplot as plt
from twitter_api import *

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

def percentage(sum, total):
    return 100 * float(sum)/float(total)

# Sentiment Statistic Output
def per(tallied_total):
    return (str(percentage(tallied_total[0],TWEET_SAMPLE_SIZE)) + '% NEGATIVE  ' 
    + str(percentage(tallied_total[1],TWEET_SAMPLE_SIZE)) + '% NEUTRAL  '
    + str(percentage(tallied_total[2],TWEET_SAMPLE_SIZE)) + '% POSITIVE')


# Returns result for data
def result(data, results):
    if data[0] > 0.60:
        results[0] += 1
        #return 'Negative'
    elif data[1] > 0.60:
        results[1] += 1
        #return 'Neutral'
    elif data[2] > 0.60: 
        results[2] += 1   
        #return 'Positive'
    else:
        return 'Not Conclusive'

# Reads and formats Tweets csv file into dataframe
df = open("csv/tweets.csv", "r")
tweets_df = pd.read_csv('csv/tweets.csv',index_col=0)
tweets_df = tweets_df.reset_index()

results = [0,0,0]
tweet_num = 0
trend_num = 0

# Inputs each tweet into roBERTa model
for tweet in tweets_df.loc[:,'Tweet']:

    input = tokenizer(format_tweet(tweet), return_tensors='pt')
    output = model(**input)
    sm = output[0][0].detach().numpy()
    score = softmax(sm).tolist()

    result(score, results)
    tweet_num += 1
    
    if tweet_num % TWEET_SAMPLE_SIZE == 0:
        tweet = api.update_status('Trend: ' + str(trend_data[trend_num][1]) + '   ' # Live tweeting format
        + per(results) + '    SAMPLE SIZE: ' + str(TWEET_SAMPLE_SIZE))

        id = tweet.id_str

        
        results = [0,0,0]




        presentage_results = [percentage(results[0],TWEET_SAMPLE_SIZE),percentage(results[0],TWEET_SAMPLE_SIZE)]

        labels = 'Negative', 'Neutral', 'Positive', 'Unclear'
        sizes = [15, 30, 45, 10]
        explode = (0.1, 0.1, 0.1, 0.1)  

        fig1, ax1 = plt.subplots()
        plt.title(str(trend_data[trend_num][1]), fontsize = 15)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        plt.savefig('graphs/' + str(trend_data[trend_num][1]) + 'data')



        trend_num +=1   

