# Import the necessary package to process data in JSON format
try:
	import json
except ImportError:
	import simplejson as json

# We use the file saved from last step as example
tweets_filename = 'twitter_stream_1000tweets.txt'
tweets_file = open(tweets_filename, "r")

for line in tweets_file:
	try:
		# Read in one line of the file, convert it into a json object 
		tweet = json.loads(line.strip())
		#print(tweet)
		if 'text' in tweet: # only messages contains 'text' field is a tweet
			print("========ID : \n", tweet['id']) # This is the tweet's id
			print("========created_at : \n", tweet['created_at']) # when the tweet posted
			print("========text : \n", tweet['text']) # content of the tweet
						
			print("========User ID: \n", tweet['user']['id']) # id of the user who posted the tweet
			print("========User Name: \n", tweet['user']['name']) # name of the user, e.g. "Wei Xu"
			print("========User screen_name: \n", tweet['user']['screen_name']) # name of the user account, e.g. "cocoweixu"

			hashtags = []
			for hashtag in tweet['entities']['hashtags']:
				hashtags.append(hashtag['text'])
			print("========Hashtags : \n", hashtags)
	except:
		# read in a line is not in JSON format (sometimes error occured)
		continue