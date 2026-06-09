import json
from collections import Counter

with open(r"E:\Projects\Columbus\data\columbus-h3 res 8-at-trips-with-additionalmetricsV2.geojson", encoding="utf-8") as f:
    data = json.load(f)

features = data["features"]
p = features[0]["properties"]
print("All property keys:", list(p.keys()))
print()

# Check every field that could be a crash rating
for key in p.keys():
    vals = [f["properties"].get(key) for f in features]
    non_null = [v for v in vals if v is not None]
    print(f"{key!r}:  {len(non_null)}/{len(vals)} non-null  |  sample non-null: {non_null[:5]}")
