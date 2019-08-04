import os
import pandas as pd
import re

stripn = lambda x: x.split('\n')[2].strip()

def proccess_meta(filename):

	with open(filename,'r') as f:
		res = f.read()

	res = [stripn(i) for i in res.split('|')]

	bookid = re.sub('[^0-9]','',filename)

	colnames = ['nameid','name','book_date',
	'city','holding_location','age','height','weight','race','sex','eyes','hair']
	res = dict(zip(colnames, res))

	res['bookid'] = bookid

	return res

data = []
for filename in os.listdir('./meta'):
	try:
		res = proccess_meta('./meta/' + filename)
		data.append(res)
	except:
		print(filename)
		continue

data = pd.DataFrame(data)
data.to_csv('./full.csv', index = False)