from summa import keywords
import json


with open("./frontend/interface/public/topic_cluster_texts.json", "r") as f:
    summaries = json.load(f)

topic_keywords = {}
for i, topic_index in enumerate(summaries):
   text = summaries[i]
   keyw = keywords.keywords(text, words=1)
   topic_keywords[f"topic_{i}"] = keyw

with open("top_keywords.json", "w") as f:
    json.dump(topic_keywords, f, indent=2)

print("write top_keywords.json")