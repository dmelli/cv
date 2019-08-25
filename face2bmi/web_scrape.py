import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import os
import re
from selenium import webdriver
from tqdm import tqdm
from datetime import datetime
from selenium.webdriver.chrome.options import Options

def _mkdir_if_not_exist(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

def _check_new_bookid(new_bookid):
	num_total = len(new_bookid)
	try:
		exist_bookid = pd.read_csv('./full.csv')['bookid'].values
	except:
		exist_bookid = []
	new_bookid = [i for i in new_bookid if i not in exist_bookid]
	num_new = len(new_bookid)
	print('{}/{} bookids are new.'.format(num_new,num_total))
	return new_bookid

def _stripn(x):

	x = x.split('\n')[2].strip()
	return x

def _scrape_page(bookid):
	"""scrape detail pages
	Args:
		@urlpage
		@bookid
	Return:

	"""
	url = urlpage + 'Details?Book_ID='+ str(bookid)

	response = requests.get(url)
	soup = BeautifulSoup(response.text, 'html.parser')

	# parse table
	tbl = soup.find('div','col-md-9').findAll('div')
	res = [i.text for i in tbl]
	with open('./meta/{}.txt'.format(bookid),'w') as f:
		f.write('|'.join(res))
 
	img_url = soup.find('div','col-md-3').find('img')['src']
	# save image
	urllib.request.urlretrieve('https://apps.polkcountyiowa.gov' + img_url, './face/'+ str(bookid)+'.jpg')

	return res

def _proccess_meta(filename):

	with open(filename,'r') as f:
		res = f.read()

	res = [_stripn(i) for i in res.split('|')]

	bookid = re.sub('[^0-9]','',filename)

	colnames = ['nameid','name','book_date',
	'city','holding_location','age','height','weight','race','sex','eyes','hair']
	res = dict(zip(colnames, res))

	res['bookid'] = bookid

	return res

def println(x, l = 50):
	pad_l = (l - len(str(x)))//2
	pad_r = l - pad_l - len(str(x))
	print('='*pad_l + str(x) + '='*pad_r)


def scrape_bookid(urlpage):
	""" scrape the bookid
	Args:
		@urlpage(str): the base url
	Return:
		bookid(list)
	"""
	# set headless browser
	options = webdriver.ChromeOptions()
	options.headless = True

	# run chrome webdriver from executable path of your choice
	driver = webdriver.Chrome(executable_path = CHROME_DRIVER,
							  options = options)

	# get web page
	driver.get(urlpage)

	print('waiting 15s for page loading')
	# sleep for 30s
	time.sleep(15)

	# find elements by xpath
	xpath = "//*[@id='DataTables_Table_0']/tbody//*[contains(@role,'row')]"
	results = driver.find_elements_by_xpath(xpath)
	print('Number of results', len(results))

	# create empty array to store data
	data = []
	# loop over results
	for result in tqdm(results):
		_, lastname, firstname, age, book_date = [i.text for i in result.find_elements_by_tag_name('td')]
		link = result.find_element_by_xpath('td/a').get_attribute('href')
		# append dict to array
		row = {}
		row['lastname'] = lastname
		row['firstname'] = firstname
		row['age'] = age
		row['book_date'] = book_date
		row['link'] = link
		data.append(row)

	driver.quit()
	df = pd.DataFrame(data)
	df['bookid'] = df.link.map(lambda i: i.split('?')[-1])\
						  .map(lambda i: i.split('=')[-1])\
						  .map(int)
	_mkdir_if_not_exist('./bookid')
	new_bookid = df.bookid.tolist()
	new_bookid = _check_new_bookid(new_bookid)
	df = df.loc[df.bookid.isin(new_bookid),:]
	df.to_csv(BOOKID_PATH, index=False)

	return new_bookid

def scrape_pages_continue(bookids):
	_mkdir_if_not_exist('./meta')
	_mkdir_if_not_exist('./face')
	existing = os.listdir('./meta')
	existing = [i.split('.')[0] for i in existing]
	bookids = [i for i in bookids if str(i) not in existing]

	data = {}
	for bookid in tqdm(bookids):
		try:
			res = _scrape_page(bookid)
			data[bookid] = res
		except:
			continue
	return data

def combine_meta():
	data = []
	for filename in os.listdir('./meta'):
		try:
			res = _proccess_meta('./meta/' + filename)
			data.append(res)
		except:
			print(filename)
			continue
	data = pd.DataFrame(data)

	return data

if __name__ == '__main__':

	# chrome drive can be downloaded from 
	# https://chromedriver.chromium.org/downloads
	# current stable version is ChromeDriver 76.0.3809.126
	# extract the chromedrive of your OS and place into the path of './chromedriver'
	CHROME_DRIVER = './chromedriver'

	# save each extraction to './bookid' directory
	BOOKID_PATH = './bookid/%s.csv'%(str(datetime.today().date()))

	# the first page for web scraping
	urlpage = 'https://apps.polkcountyiowa.gov/PolkCountyInmates/CurrentInmates/' 

	println(' fetch new bookid list ')
	if os.path.exists(BOOKID_PATH):
		print('> load from existing bookid list [%s]'%(BOOKID_PATH))
		new_bookid = pd.read_csv(BOOKID_PATH)['bookid'].tolist()
		print('> found %d bookid'%(len(new_bookid)))
	else:
		new_bookid = scrape_bookid(urlpage)

	println(' scrape new pages ')
	_ = scrape_pages_continue(new_bookid)

	println(' combine data ')
	data = combine_meta()
	data.to_csv('./full.csv', index = False)


