your_information = {'YOUR ACCESS TOKEN': '3188918876-RyKM2FcoU0iAVZmHrucR6qDFr7WnghxlD3BHcTf', 'YOUR ACCESS TOKEN SECRET': '5d6qCr1dqAUSyDGViRZxzhs8Ph1C0LrTZLYaa85qlGkug',
					'YOUR API KEY': 'uxpGD2hoIzHhe0Kijzj88dojb','ENTER YOUR API SECRET': '2L51B1x63QtQ7MOts2dhGbpC0qRu5SItMSfJKMcV5xGef3WUZi'}

# Import the necessary package to process data in JSON format
try:
	import json
except ImportError:
	import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = your_information['YOUR ACCESS TOKEN']
ACCESS_SECRET = your_information['YOUR ACCESS TOKEN SECRET']
CONSUMER_KEY = your_information['YOUR API KEY']
CONSUMER_SECRET = your_information['ENTER YOUR API SECRET']


oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter = Twitter(auth=oauth)
# Alternatively, you can search with more parameters (see a full list here). For example, search for 10 latest tweets about “#nlproc” in English:
#iterator = twitter.search.tweets(q='#nlproc')
iterator = twitter.search.tweets(q='#nlproc', result_type='recent', lang='en', count=1)
print(iterator)
for tweet in iterator:
	print(tweet['id_str'], '\n', tweet['text'])


'''
# Trends API
# Twitter provides global trends and as well as localized tweets.
# The easiest and best way to see what trends are available and the place ids (which you will need to know to query localized trends or tweets), 
# is by using this commend to request worldwide trends:

# Get all the locations where Twitter provides trends service
world_trends = twitter.trends.available(_woeid=1)

# After you know the ids for the places you are interested in, you can get the local trends like this:
# The places ids are WOEIDs (Where On Earth ID), which are 32-bit identifiers provided by Yahoo! GeoPlanet project. And yes! Twitter is very international.
# Get all (it's always 10) trending topics in San Francisco (its WOEID is 2487956)
sfo_trends = twitter.trends.place(_id = 2487956)
print(json.dumps(sfo_trends, indent=4))


# User API
# Get a list of followers of a particular user
twitter.followers.ids(screen_name="cocoweixu")
# Get a particular user's timeline (up to 3,200 of his/her most recent tweets)
twitter.statuses.user_timeline(screen_name="billybob")
#twitter.application.rate_limit_status()
'''