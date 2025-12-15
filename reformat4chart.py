import json, math
from datetime import datetime, timedelta
from collections import defaultdict

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

with open("clusters.json", "r") as f:
    clusters = json.load(f)

with open("dataset.json", "r") as f:
    data = json.load(f)

# ---- Extract timestamps dataset ----------------
def extract_timestamps(data_obj):
    #check if its a dictionary
    if isinstance(data_obj, dict):

        #check ig key "timestamp" exists
        if "timestamp" in data_obj:

            #get the timestamp column
            ts_col = data_obj["timestamp"]

            #check if the column is a dict which it is but good to do
            if isinstance(ts_col, dict):
                #sort keys my numeric value and return
                return [ts_col[k] for k in sorted(ts_col.keys(), key=lambda x: int(x))]

            raise TypeError("Found data['timestamp'] but it's not a dict")


timestamps = extract_timestamps(data)

# Parse times
start_time = datetime.strptime(timestamps[0], DATE_FORMAT)
end_time = datetime.strptime(timestamps[-1], DATE_FORMAT)
start_week_monday = start_time.date() - timedelta(days=start_time.date().weekday())

# Count posts per (week, topic)
#dictionary of int keys (week) to dictionary of int keys (topic index) to ints (counts)
#check drawing onenote
counts = defaultdict(lambda: defaultdict(int))

#go through topics
for topic_index, cluster in enumerate(clusters):
    for row_index in cluster:
        ts = datetime.strptime(timestamps[row_index], DATE_FORMAT)
        week = math.floor((ts.date() - start_week_monday).days / 7)
        counts[week][topic_index] += 1

num_topics = len(clusters)
max_week = max(counts.keys()) #the num of weeks

chart_rows = []

for week in range(max_week + 1):
    week_start = start_week_monday + timedelta(days=7 * week)

    #turn date to string, add date of start of week and the week index
    row = {"week": week_start.isoformat(), "__weekIndex": week} 

    #for each week for each topic add the count, the row is a dictionary
    #the key is topic_{topic_index} and the value is the count
    for topic_index in range(num_topics):
        row[f"topic_{topic_index}"] = counts[week].get(topic_index, 0)
    
    chart_rows.append(row)

with open("weekly_topic_counts.json", "w") as f:
    json.dump(chart_rows, f, indent=2)

print("Wrote weekly_topic_counts.json")

weeks = [row["week"] for row in chart_rows]
with open("weeks.json", "w") as f:
    json.dump(weeks, f, indent=2)

print("Wrote weeks.json")
