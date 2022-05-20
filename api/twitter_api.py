import tweepy, configparser, pandas as pd, os
import config

TWEET_SAMPLE_SIZE = 200

# compile_trendss trend_data about trending topics into list 
def compile_trends(trends, trend_data):
    for trend in trends[0]['trends']:
        if trend['tweet_volume'] is not None and trend['tweet_volume'] > 10000:
            trend_data.append((trend['tweet_volume'],trend['name'],trend['url']))

    trend_data.sort(reverse = True, key = lambda x: x[0])
    return trend_data

# compiles tweets about a trend
def compile_tweets(trend, tweet_data):
    tweets = tweepy.Cursor(api.search_tweets, q = trend[1] + '-filter:retweets -filter:replies',lang = 'en' 
    ,count = TWEET_SAMPLE_SIZE, tweet_mode = 'extended').items(TWEET_SAMPLE_SIZE)

    for tweet in tweets:
        tweet_data.append((tweet.created_at,tweet.user.screen_name, tweet.full_text))
    
# read Configs from ini file
config = configparser.ConfigParser()
config.read('api/config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# tweepy authentication
auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# trends based on location
trends = api.get_place_trends(id = 23424775)


# trend_dataframe lists
trends_column = ['tweet_volume','Topic','URL']
trend_data = []
tweet_column = ['Time_tweeted','User','Tweet']
tweet_data = []
compile_trends(trends,trend_data)

# top 10 tweeted trending
trend_data = trend_data[0:3]

# converts trend_data into csv file
if os.path.exists('csv/trending_topics.csv'):
    os.remove('csv/trending_topics.csv') #removes old trending csv file
trend_dataframe = pd.DataFrame(trend_data, columns=trends_column)
trend_dataframe.to_csv('csv/trending_topics.csv')


if os.path.exists('csv/tweets.csv'):
    os.remove('csv/tweets.csv') #removes old tweets csv file

# compiles tweets into csv file
end = TWEET_SAMPLE_SIZE + 1
for trend in trend_data:
    compile_tweets(trend,tweet_data)
    tweet_dataframe = pd.DataFrame(tweet_data,index=range(1,end) ,columns = tweet_column)
    end += TWEET_SAMPLE_SIZE

tweet_dataframe.to_csv('csv/tweets.csv')
