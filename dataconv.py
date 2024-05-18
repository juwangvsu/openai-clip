import pandas as pd
import config as CFG
df = pd.read_csv(f"{CFG.captions_path}/captions.txt")
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv(f"{CFG.captions_path}/captions.csv", index=False)
