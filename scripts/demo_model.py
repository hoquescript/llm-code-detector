import pandas as pd

df = pd.read_json("../data/codegptsensor/python/train.jsonl", lines=True)
print(df.head(3))