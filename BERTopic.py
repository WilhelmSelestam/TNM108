import pandas as pd
from summa import keywords
import re
import spacy
from bertopic import BERTopic
nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('Social_Media_Engagement_Dataset.csv')

#print(data[['text_content', 'timestamp']])

data = data[['timestamp', 'day_of_week', 'location', 'text_content', 'engagement_rate']]

data['text_content'] = data['text_content'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
data['text_content'] = data['text_content'].str.lower()

data = data.sort_values(by=['timestamp'])

timestamps = data['timestamp'].to_list()
text = data['text_content'].to_list()

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(text)

topics_over_time = topic_model.topics_over_time(text, timestamps, nr_bins=20)

fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
fig.show()