import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy
from scipy.special import softmax
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

def format_data(datapoint):
    result = []
    result.append('Negative: ' + str(round(datapoint[0]*100, 1)) + '%')
    result.append('Neutral: ' + str(round(datapoint[1]*100, 1)) + '%')
    result.append('Positive: ' + str(round(datapoint[2]*100, 1)) + '%')
    return result

df = open("csv/tweets.csv", "r")
tweets_df = pd.read_csv('csv/tweets.csv',index_col=0)
tweets_df = tweets_df.reset_index()

# test = 'questionable'
# out = tokenizer(test, return_tensors='pt')
# print(out)
# print(tokenizer.convert_ids_to_tokens([0, 40018, 868, 2]))

for tweet in tweets_df.loc[:,'Tweet']:
    input = tokenizer(format_tweet(tweet), return_tensors='pt')
    output = model(**input)
    sm = output[0][0].detach().numpy()
    score = softmax(sm).tolist()
    print(format_data(score))