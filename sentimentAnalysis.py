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
import json
from sklearn import metrics


data = pd.read_csv('Social_Media_Engagement_Dataset.csv')

data = data[[ 'text_content','sentiment_score','sentiment_label']]

data['text_content'] = data['text_content'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
data['text_content'] = data['text_content'].str.lower()


posts = []
sentiment_scores = []
sentiment_labels = []

for row in data.itertuples(index=False):
    posts.append(row.text_content)
    sentiment_scores.append(row.sentiment_score)
    sentiment_labels.append(row.sentiment_label)

print(sentiment_labels[:10])


#------Getting sentiment using transformers-------------
# sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# results = sentiment_pipeline(posts)

# scores = []
# score_labels = []

# for res in results:
#     score_val = res['score']
#     score_label = res['label']
#     if res['label'] == 'NEGATIVE':
#         score_val = -score_val

#     scores.append(score_val)
#     score_labels.append(score_label)



# with open('scores.json', 'w') as f:
#     json.dump(scores, f)

# with open('labels.json', 'w') as f:
#     json.dump(score_labels, f)
#------end Getting sentiment using transformers-------------

with open('scores.json', 'r') as f:
    scores = json.load(f)

with open('labels.json', 'r') as f:
    score_labels = json.load(f)


print("our sentiment:",  score_labels[:10])
correct = 0
total = 0
for i in range(len(sentiment_labels)):
    if str(sentiment_labels[i]).lower() == 'neutral':
        continue
    total += 1
    if str(sentiment_labels[i]).lower() == str(score_labels[i]).lower():
        correct += 1
accuracy = correct / total * 100
print(f'Accuracy of transformer-based sentiment analysis: {accuracy:.2f}%')

confusion_mtx = metrics.confusion_matrix(sentiment_labels, score_labels)
print("Confusion Matrix:")
print(confusion_mtx)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_mtx)
cm_display.plot()
plt.show()
total_diff = 0
for i in range(len(sentiment_scores)):
    diff = abs(sentiment_scores[i] - scores[i])
    total_diff += diff

avg_diff = total_diff / len(sentiment_scores)
print(f'Average difference in sentiment scores: {avg_diff:.4f}')
