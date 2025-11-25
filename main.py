
import pandas as pd

data = pd.read_csv('Social_Media_Engagement_Dataset.csv')

print(data[['text_content', 'timestamp']])

data = data.sort_values(by=['timestamp'])

print(data[['text_content', 'timestamp']])

