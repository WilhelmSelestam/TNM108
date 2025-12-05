import pandas as pd
# from summa import keywords
# import nltk
#from nltk.stem import PorterStemmer
#import spacy
from datetime import datetime, timedelta
import math
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import find_peaks
from collections import defaultdict
import textwrap
from transformers import pipeline

#nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('data.csv').head(400)

#date format things
date_format = '%Y-%m-%d %H:%M:%S'

#print(data['keywords'])

model = SentenceTransformer('all-MiniLM-L6-v2')

embedded_posts = []
posts = []

for row in data.itertuples(index=False):
    embedded_posts.append(model.encode(row.text_content))
    posts.append(row.text_content)


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
topicSeries = [[]]

start_time = datetime.strptime(data['timestamp'].iloc[0], date_format)
end_time = datetime.strptime(data['timestamp'].iloc[-1], date_format)

time_span = end_time - start_time

nr_of_weeks = math.ceil(time_span.days / 7)

for week in range (0, nr_of_weeks):
    topicSeries.append([])

#print(nr_of_weeks)

current_week = 0
week_length = timedelta(days=7)

start_time_first_day_of_week = start_time.date() - timedelta(days=start_time.date().weekday())
counter = 0
for row in data.itertuples(index=False):

    current_date = datetime.strptime(row.timestamp, date_format)
    #print()  
    #print(type(row.keywords))
    # keywords2 = row.keywords.replace('[','')
    # keywords2 = keywords2.replace(']','')
    # keywords2 = keywords2.replace("'",'')
    # keywords2 = keywords2.replace(',','').split()
    #print(embedded_posts[counter])
    if current_date.date() < start_time_first_day_of_week + week_length * (current_week + 1):
        # for keyword in keywords2:
        #     topicSeries[current_week][keyword] = topicSeries[current_week].get(keyword, 0) + 1
            #topicSeries[current_week].append(row)
            #print(keyword)
        topicSeries[current_week].append(embedded_posts[counter])
    else:
        #timeSeries.append(str(first_day_of_week) + " - " + str(current_date))
        current_week += 1
        #topicSeries.append([row])
        #first_day_of_week = datetime.strptime(row.timestamp, date_format) - timedelta(datetime.strptime(row.timestamp, date_format).weekday())
        # for keyword in keywords2:
        #     topicSeries[current_week][keyword] = topicSeries[current_week].get(keyword, 0) + 1
        topicSeries[current_week].append(embedded_posts[counter])

    counter+=1

first_day_of_week = start_time.date() - timedelta(days=start_time.date().weekday())
while(first_day_of_week < end_time.date()):
    timeSeries.append(str(first_day_of_week) + " - " + str(first_day_of_week + timedelta(days=6)))
    first_day_of_week = first_day_of_week + timedelta(days=7)

keywords = []
topic_clusters = []
counter = 0

# for i, week in enumerate(timeSeries):
 
clusters = util.community_detection(
        embedded_posts, 
        threshold=0.6, 
        min_community_size=1
)
#topic_clusters.append(clusters)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def chunk_text(text, max_chars=2000):
    """Split long text into smaller pieces."""
    for i in range(0, len(text), max_chars):
        yield text[i:i+max_chars]

topic_cluster_texts = []

for cluster in clusters:
    text = ""
    for row_index in cluster:
        text += posts[row_index] + " "  # add space between posts

    # 1) Summarize each chunk
    partial_summaries = []
    for chunk in chunk_text(text):
        summary = summarizer(
            chunk,
            max_length=80,    # length of each chunk summary
            min_length=20,
            do_sample=False
        )[0]["summary_text"]
        partial_summaries.append(summary)

combined = " ".join(partial_summaries)
final_summary = summarizer(
    combined,
    max_length=50,
    min_length=10,
    do_sample=False
)[0]["summary_text"]

topic_cluster_texts.append(final_summary)
print(topic_cluster_texts[0])

    # for topics in topicSeries[i]:
        
        # grouped_posts = [posts[cluster[0]] for cluster in clusters]
        # print(grouped_posts)


        # keywords.append(topics)
            
# keywords = list(set(keywords)) 

# keywordsDict = dict()
# for i, keyword in enumerate(keywords):
#     keywordsDict.update({keyword: i})

# # print(keywordsDict)
# topicMatrix = [[0 for _ in range(len(timeSeries))] for _ in range(len(keywords))]

# for i, week in enumerate(timeSeries):
#     for topics in topicSeries[i]:
#         topicMatrix[keywordsDict[topics]][i] = topicSeries[i].get(topics)
        
        
        
        
        
        
        
# for i, row in enumerate(topicMatrix):    
    
#     peaks, _ = find_peaks(row, height=3, width=1)
#     print(peaks, keywords[i])
    
    
    # diff = np.diff(row)
    # row2 = row[:-1]
    # for i, r in enumerate(row2):
    #     if r < 1e-9:
    #         row2[i] = 1e-9
        
    # pct_change = diff / row2

    
    # spikes = np.where(pct_change > 5000)[0] + 1
    # # print(pct_change)
    # print("spike weeks: ", spikes)
    
    
    # mean = np.mean(row)
    # std = np.std(row)
    # if std != 0:
    #     z_scores = (row - mean) / std
    #     spikes = np.where(z_scores > 2)[0]
        
    #     # print(len(spikes))
    
    #     if len(spikes) == 0:
    #         # data.append(row)
    #         topicMatrix[i] = [0 for _ in range(len(row))]
    #     # print(row)
    # else:
    #     topicMatrix[i] = [0 for _ in range(len(row))]
    
    
# print(topicMatrix) 
    
# print(topicSeries)
# print("\n")
# print(keywords)
# print("\n")
# print(keywordsDict)
# print("\n")
# print(topicMatrix)

# x = timeSeries
# # bottom_values = [0] * len(x)

# for i, row in enumerate(topicMatrix):
#     plt.plot(x, row)
#     # bottom_values = [b + v for b, v in zip(bottom_values, row)]

# plt.legend()
# plt.show()
