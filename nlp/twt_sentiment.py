import pandas as pd

#df = open("csv/tweets.csv", "r")
tweets_df = pd.read_csv('csv/tweets.csv',index_col=0)
tweets_df = tweets_df.reset_index()

tweet_number = 0

#print(tweets_df)
index = 1
for tweet in tweets_df.iterrows():
    print(tweet[1])
    index +=1