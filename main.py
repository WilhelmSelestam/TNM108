import pandas as pd
# from summa import keywords
# import nltk
#from nltk.stem import PorterStemmer
#import spacy
from datetime import datetime, timedelta
import math
from sentence_transformers import SentenceTransformer, util
#import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import find_peaks
from collections import defaultdict
import textwrap
from transformers import pipeline
import json

#nlp = spacy.load("en_core_web_sm")

print("Loading data...")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#data = pd.read_csv('data.csv').head(20)

data = pd.read_csv('Social_Media_Engagement_Dataset.csv')

data = data[['timestamp', 'day_of_week', 'location', 'text_content', 'engagement_rate']]

data['text_content'] = data['text_content'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
data['text_content'] = data['text_content'].str.lower()

data = data.sort_values(by=['timestamp'])

#date format things
date_format = '%Y-%m-%d %H:%M:%S'

#print(data['keywords'])

model = SentenceTransformer('all-MiniLM-L6-v2')

embedded_posts = []
posts = []
timestamps = []

for row in data.itertuples(index=False):
    embedded_posts.append(model.encode(row.text_content))
    posts.append(row.text_content)
    timestamps.append(row.timestamp)

timeSeries = []
topicSeries = [[]]

start_time = datetime.strptime(data['timestamp'].iloc[0], date_format)
end_time = datetime.strptime(data['timestamp'].iloc[-1], date_format)

time_span = end_time - start_time

nr_of_weeks = math.ceil(time_span.days / 7)

for week in range (0, nr_of_weeks):
    topicSeries.append([])

current_week = 0
week_length = timedelta(days=7)

start_time_first_day_of_week = start_time.date() - timedelta(days=start_time.date().weekday())
counter = 0
for row in data.itertuples(index=False):

    current_date = datetime.strptime(row.timestamp, date_format)
    if current_date.date() < start_time_first_day_of_week + week_length * (current_week + 1):
        topicSeries[current_week].append(embedded_posts[counter])
    else:
        current_week += 1
        topicSeries[current_week].append(embedded_posts[counter])

    counter+=1

first_day_of_week = start_time.date() - timedelta(days=start_time.date().weekday())
while(first_day_of_week < end_time.date()):
    timeSeries.append(str(first_day_of_week) + " - " + str(first_day_of_week + timedelta(days=6)))
    first_day_of_week = first_day_of_week + timedelta(days=7)

keywords = []
topic_clusters = []
counter = 0
 
clusters = util.community_detection(
        embedded_posts, 
        threshold=0.6, 
        min_community_size=1
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# sentiment_pipeline = pipeline(
#     "sentiment-analysis", 
#     model="distilbert-base-uncased-finetuned-sst-2-english"
# )

# def chunk_text(text, max_chars=1024):
#     """Split long text into smaller pieces."""
#     for i in range(0, len(text), max_chars):
#         yield text[i:i+max_chars]

# topic_cluster_texts = []

# for cluster in clusters:
#     text = ""
#     for row_index in cluster:
#         text += posts[row_index] + " "  # add space between posts
        

#     # 1) Summarize each chunk
#     partial_summaries = []
#     for chunk in chunk_text(text):
#         summary = summarizer(
#             chunk,
#             max_length=80,    # length of each chunk summary
#             min_length=20,
#             do_sample=False
#         )[0]["summary_text"]
#         partial_summaries.append(summary)

#         combined = " ".join(partial_summaries)
#     final_summary = summarizer(
#         combined,
#         max_length=50,
#         min_length=10,
#         do_sample=False
#     )[0]["summary_text"]

#     topic_cluster_texts.append(final_summary)

# print(topic_cluster_texts[0])

stage1_inputs = []       # List of all text chunks from all clusters
chunk_to_cluster_map = [] # Keeps track of which cluster a chunk belongs to

def chunk_text(text, max_chars=1024):
    """Split long text into smaller pieces."""
    for i in range(0, len(text), max_chars):
        yield text[i:i+max_chars]

# Iterate through clusters to prepare inputs
for cluster_idx, cluster in enumerate(clusters):
    # Combine posts in the cluster
    full_text = ""
    for row_index in cluster:
        full_text += posts[row_index] + " "
    
    # split into chunks and add to our flat list
    for chunk in chunk_text(full_text):
        stage1_inputs.append(chunk)
        chunk_to_cluster_map.append(cluster_idx)

print(f"Total chunks to summarize: {len(stage1_inputs)}")

# --- STAGE 1: BATCH GENERATE PARTIAL SUMMARIES ---
# This runs the model on the massive list of chunks
print("Running Stage 1 summarization (Partial)...")
stage1_results = summarizer(
    stage1_inputs, 
    max_length=80, 
    min_length=20, 
    do_sample=False, 
    batch_size=8,
    truncation=True
)

# --- RE-GROUPING ---
# Group the partial summaries back to their respective clusters
cluster_partial_summaries = defaultdict(list)
for i, result in enumerate(stage1_results):
    cluster_idx = chunk_to_cluster_map[i]
    cluster_partial_summaries[cluster_idx].append(result['summary_text'])

# --- STAGE 2: BATCH GENERATE FINAL SUMMARIES ---
# Now we combine the partial summaries and do one final pass
print("Running Stage 2 summarization (Final)...")

stage2_inputs = []
# Ensure we process clusters in order (0, 1, 2...)
sorted_cluster_indices = sorted(cluster_partial_summaries.keys())

for cluster_idx in sorted_cluster_indices:
    # Join all partial summaries for this cluster
    combined_text = " ".join(cluster_partial_summaries[cluster_idx])
    if len(combined_text) > 4000:
        combined_text = combined_text[:4000]
    stage2_inputs.append(combined_text)

# Run final batch summarization
stage2_results = summarizer(
    stage2_inputs,
    max_length=50,
    min_length=10,
    do_sample=False,
    batch_size=8,
    truncation=True
)

# Extract final texts
topic_cluster_texts = [res['summary_text'] for res in stage2_results]

sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0
)

max_week_number = math.floor((end_time.date() - start_time_first_day_of_week).days / 7)
organized_data = defaultdict(lambda: defaultdict(list))

for i, cluster in enumerate(clusters):
    for row_index in cluster:
        current_timestamp = datetime.strptime(timestamps[row_index], date_format)
        current_text = posts[row_index]
        
        week_number = math.floor((current_timestamp.date() - start_time_first_day_of_week).days / 7)
        
        # Store individual posts in a list rather than concatenating
        organized_data[week_number][i].append(current_text)

# 4. Run Sentiment Analysis and Calculate Averages
weekly_sentiment_scores = []

print("Running sentiment analysis on batches...")

# Initialize the result structure
for week in range(max_week_number + 1):
    current_week_scores = []
    
    for cluster_idx in range(len(clusters)):
        posts_in_cluster = organized_data[week][cluster_idx]
        
        if not posts_in_cluster:
            # No posts for this cluster in this week
            current_week_scores.append(None) 
            continue
            
        # Run pipeline on the LIST of posts (Batch Processing)
        # batch_size=16 or 32 is usually good for standard GPUs
        results = sentiment_pipeline(posts_in_cluster, batch_size=16, truncation=True, max_length=512)
        
        # Convert results to a numerical score
        # Logic: POSITIVE = score, NEGATIVE = -score
        scores = []
        for res in results:
            score_val = res['score']
            if res['label'] == 'NEGATIVE':
                score_val = -score_val
            scores.append(score_val)
            
        # Calculate average for the week
        avg_score = np.mean(scores)
        current_week_scores.append(avg_score)
        
    weekly_sentiment_scores.append(current_week_scores)

#print(sentimented_posts)

data.to_json('dataset.json', indent=4)

with open('topic_cluster_texts.json', 'w') as f:
    json.dump(topic_cluster_texts, f)

with open('clusters.json', 'w') as f:
    json.dump(clusters, f)

with open('sentimented_posts_per_week_per_cluster.json', 'w') as f:
    json.dump(weekly_sentiment_scores, f)


#print(weekly_texts)
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
