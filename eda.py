import pandas as pd
import os

data_dir = os.path.expanduser("~/SIH/data/cic-ids2018")
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

for fname in files:
    df = pd.read_csv(os.path.join(data_dir, fname), usecols=['Label'], low_memory=False)
    df = df[df['Label'] != 'Label']
    print(f"\n{fname}")
    print(df['Label'].value_counts().to_string())
