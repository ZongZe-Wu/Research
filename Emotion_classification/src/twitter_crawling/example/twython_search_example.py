from twython import Twython


your_information = {'YOUR ACCESS TOKEN': '3188918876-RyKM2FcoU0iAVZmHrucR6qDFr7WnghxlD3BHcTf', 'YOUR ACCESS TOKEN SECRET': '5d6qCr1dqAUSyDGViRZxzhs8Ph1C0LrTZLYaa85qlGkug',
					'YOUR API KEY': 'uxpGD2hoIzHhe0Kijzj88dojb','ENTER YOUR API SECRET': '2L51B1x63QtQ7MOts2dhGbpC0qRu5SItMSfJKMcV5xGef3WUZi'}
TWITTER_APP_KEY = your_information['YOUR API KEY']
TWITTER_APP_KEY_SECRET = your_information['ENTER YOUR API SECRET']
TWITTER_ACCESS_TOKEN = your_information['YOUR ACCESS TOKEN']
TWITTER_ACCESS_TOKEN_SECRET = your_information['YOUR ACCESS TOKEN SECRET']

t = Twython(app_key=TWITTER_APP_KEY, 
			app_secret=TWITTER_APP_KEY_SECRET, 
			oauth_token=TWITTER_ACCESS_TOKEN, 
			oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

search = t.search(q='#happy', result_type='mixed', lang='en', count=100)   #**supply whatever query you want here**

tweets = search['statuses']

for tweet in tweets:
	print(tweet['id_str'], '\n', tweet['text'], '\n\n')
print('amount : ', len(tweets))