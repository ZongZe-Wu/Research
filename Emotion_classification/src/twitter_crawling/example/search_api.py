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