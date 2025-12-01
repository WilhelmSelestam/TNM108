import pandas as pd
from summa import keywords
import nltk
from nltk.stem import PorterStemmer
import spacy
from datetime import datetime, timedelta
import math
from sentence_transformers import SentenceTransformer, util

#nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('data.csv').head(35)

#date format things
date_format = '%Y-%m-%d %H:%M:%S'

#print(data['keywords'])

#embedding
#model = SentenceTransformer('all-MiniLM-L6-v2')

# keywords = set()
# for row in data.itertuples(index=False):
#     for keyword in row.keywords:
#         #print(keyword)
#         keywords.add(keyword)

# keywords = list(keywords)
# print(keywords[95])
  
# keywords = [
#     "buy car", "purchase automobile", "buy vehicle", 
#     "apple", "banana", "buy cars", "apple fruit"
# ]
# embeddings = model.encode(keywords)

# clusters = util.community_detection(
#     embeddings, 
#     threshold=0.75, 
#     min_community_size=1
# )

# print(clusters)

# filtered_keywords = [keywords[cluster[0]] for cluster in clusters]

# print(filtered_keywords)

timeSeries = []
topicSeries = [{}]

start_time = datetime.strptime(data['timestamp'].iloc[0], date_format)
end_time = datetime.strptime(data['timestamp'].iloc[-1], date_format)

time_span = end_time - start_time

nr_of_weeks = math.ceil(time_span.days / 7)

for week in range (0, nr_of_weeks):
    topicSeries.append({})

#print(nr_of_weeks)

current_week = 0
week_length = timedelta(days=7)

start_time_first_day_of_week = start_time.date() - timedelta(days=start_time.date().weekday())

for row in data.itertuples(index=False):

    current_date = datetime.strptime(row.timestamp, date_format)
    #print()  
    #print(type(row.keywords))
    keywords2 = row.keywords.replace('[','')
    keywords2 = keywords2.replace(']','')
    keywords2 = keywords2.replace("'",'')
    keywords2 = keywords2.replace(',','').split()
    
    if current_date.date() < start_time_first_day_of_week + week_length * (current_week + 1):
        for keyword in keywords2:
            topicSeries[current_week][keyword] = topicSeries[current_week].get(keyword, 0) + 1
            #topicSeries[current_week].append(row)
            print(keyword)
    else:
        #timeSeries.append(str(first_day_of_week) + " - " + str(current_date))
        current_week += 1
        #topicSeries.append([row])
        #first_day_of_week = datetime.strptime(row.timestamp, date_format) - timedelta(datetime.strptime(row.timestamp, date_format).weekday())
        for keyword in keywords2:
            topicSeries[current_week][keyword] = topicSeries[current_week].get(keyword, 0) + 1


first_day_of_week = start_time.date() - timedelta(days=start_time.date().weekday())
while(first_day_of_week < end_time.date()):
    timeSeries.append(str(first_day_of_week) + " - " + str(first_day_of_week + timedelta(days=6)))
    first_day_of_week = first_day_of_week + timedelta(days=7)

print(timeSeries)

