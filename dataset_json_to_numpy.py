import json
import numpy as np
import einops
from pathlib import Path

data = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': []}

jsondir = Path("./dataset_json/")
for json_path in jsondir.iterdir():
    if not json_path.parts[-1].endswith(".json"):
        continue
    
    label = json_path.parts[-1][0].upper()

    with open(json_path, 'r') as f:
        sample = json.load(f)
        sample = sample['payload']['values']
        sample = np.array(sample)
        data[label].append(sample)



for letter in data:
    minimum = min(x.shape[0] for x in data[letter])
    maximum = max(x.shape[0] for x in data[letter])
    print(letter)
    print(f"{minimum=}")
    print(f"{maximum=}")

    # crop to 132 elements. remove if size < 132
    data[letter] = list(x[:132] for x in data[letter] if x.shape[0] >= 132)
    data[letter], _ = einops.pack(data[letter], "* time col")
    print(f"sample count = {data[letter].shape[0]}")


OUT_PATH = "ABCDEF.npz"
np.savez(OUT_PATH, **data)
print(f"saved to {OUT_PATH}")
