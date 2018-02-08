
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
twitter_stream = TwitterStream(auth=oauth)

# First, you can set different parameters (see here for a complete list) to define what data to request.
# For example, you can track certain tweets by specifying keywords or location or language etc.
# The following code will get the tweets in English that include the term “Google”:
#iterator = twitter_stream.statuses.filter(track="Google", language="en")
#iterator = twitter_stream.statuses.filter(language="en")

# Get a sample of the public data following through Twitter
iterator = twitter_stream.statuses.sample(language="en")

# For conducting research on Twitter data, you usually only need to use “public streams” to collect data. In case you do need to use other streams, here is how to specify it:
#twitter_userstream = TwitterStream(auth=oauth, domain='userstream.twitter.com')
# Print each tweet in the stream to the screen 
# Here we set it to stop after getting 1000 tweets. 
# You don't have to set it to stop, but can continue running 
# the Twitter API to collect data for days or even longer. 
tweet_count = 10
for tweet in iterator:
	tweet_count -= 1
	# Twitter Python Tool wraps the data returned by Twitter 
	# as a TwitterDictResponse object.
	# We convert it back to the JSON format to print/score
	print(json.dumps(tweet))  
	
	# The command below will do pretty printing for JSON data, try it out
	#print(json.dumps(tweet, indent=4))
	   
	if tweet_count <= 0:
		break 

