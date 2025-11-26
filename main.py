import pandas as pd
from summa import keywords
import nltk
from nltk.stem import PorterStemmer
import spacy

ps = PorterStemmer()

nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('Social_Media_Engagement_Dataset.csv')

#print(data[['text_content', 'timestamp']])

data = data[['timestamp', 'day_of_week', 'location', 'text_content', 'engagement_rate']].head(20)

data['text_content'] = data['text_content'].str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
data['text_content'] = data['text_content'].str.lower()

data = data.sort_values(by=['timestamp'])

def extract_keywords(text):
    if pd.isna(text) or not str(text).strip():
        return ''
    try:
        return keywords.keywords(str(text), words=3)
    except Exception:
        return ''

def lemming(keywords):
  #print(type(keyword))
  tokens = keywords.replace('\n',' ').split()
  #keywords.split(' ')
  stems = []


  for token in tokens:
      word = nlp(token)
      print(word)
      for w in word:
        stems.append(w.lemma_)
    
  #print(stems)

  return stems

doc = nlp('ran')

for d in doc:
    print(d,d.lemma_)

data['keywords'] = data['text_content'].apply(extract_keywords) #borde vara array av stringar

data['keywords'] = data['keywords'].apply(lemming)

print(data['keywords'])