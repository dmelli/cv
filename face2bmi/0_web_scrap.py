import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import os

bookids = pd.read_csv('./BookID.csv').iloc[:,0].values

base_url = 'https://apps.polkcountyiowa.gov'

def scrap_page(bookid):

	url = base_url + '/PolkCountyInmates/CurrentInmates/Details?Book_ID='+ str(bookid)

	response = requests.get(url)
	soup = BeautifulSoup(response.text, 'html.parser')

	# parse table
	tbl = soup.find('div','col-md-9').findAll('div')
	res = [i.text for i in tbl]
	with open('./meta/{}.txt'.format(bookid),'w') as f:
		f.write('|'.join(res))
 
	img_url = soup.find('div','col-md-3').find('img')['src']
	# save image
	urllib.request.urlretrieve(base_url + img_url, './face/'+ str(bookid)+'.jpg')
	print('{}.jpg image saved'.format(bookid))

	return res


existing = os.listdir('./meta')
existing = [i.split('.')[0] for i in existing]
bookids = [i for i in bookids if str(i) not in existing]

i = 0
total = len(bookids)
data = {}
for bookid in bookids	:
	#time.sleep(1)
	i += 1
	print('scraping {}/{}'.format(i, total))
	try:
		res = scrap_page(bookid)
		data[bookid] = res
	except:
		continue
	
	

with open('/Users/chaoran/Desktop/face_image/full.pickle','wb') as f:
	pickle.dump(data,f)

print('full data pickle saved')
