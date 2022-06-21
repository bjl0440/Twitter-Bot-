import tweepy, configparser, pandas as pd, os

# Global vars
TRENDS = 3
TWEET_SAMPLE_SIZE = 20


# compiles trends trend_data about trending topics into list 
def compile_trends(trends, trend_data):
    for trend in trends[0]['trends']:
        if trend['tweet_volume'] is not None and trend['tweet_volume'] > 10000:
            trend_data.append((trend['tweet_volume'],trend['name'],trend['url']))

    trend_data.sort(reverse = True)
    return trend_data

# compiles tweets about a trend
def compile_tweets(trend, tweet_data):
    tweets = tweepy.Cursor(api.search_tweets, q = trend[1] + '-filter:retweets -filter:replies',lang = 'en' 
    ,count = TWEET_SAMPLE_SIZE, tweet_mode = 'extended').items(TWEET_SAMPLE_SIZE)

    for tweet in tweets:
        tweet_data.append((tweet.created_at,tweet.user.screen_name, tweet.full_text))

# compiles news articles
# read Configs from ini file
config = configparser.ConfigParser()
config.read('api/config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# tweepy authentication
auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

#==============================================================================================================

# trends based on location - Canada
trends = api.get_place_trends(id = 23424775)

# trend_dataframe lists
trends_column = ['tweet_volume','Topic','URL']
trend_data = []

compile_trends(trends,trend_data)

# top number of tweeted trending
trend_data = trend_data[0:TRENDS]

# Removes previous csv files
path = r'C:\Users\bjl04\Desktop\Twitter Bot\csv\\'
for file_name in os.listdir(path):
    file = path + file_name
    if os.path.isfile(file):
        os.remove(file)

trend_dataframe = pd.DataFrame(trend_data, columns=trends_column)
trend_dataframe.to_csv('csv/trending_topics.csv')

# compiles tweets into csv file
start = 1
end = TWEET_SAMPLE_SIZE + 1
trend_index = 0

tweet_column = ['Time_tweeted','User','Tweet']
for trend in trend_data:
    tweet_data = []
    print(trend)
    compile_tweets(trend,tweet_data)
    tweet_dataframe = pd.DataFrame(tweet_data,index=range(start,end) ,columns = tweet_column)
    end += TWEET_SAMPLE_SIZE
    start += TWEET_SAMPLE_SIZE
    tweet_dataframe.to_csv('csv/' + str(trend_data[trend_index][1]) + '_tweets.csv')
    trend_index += 1
