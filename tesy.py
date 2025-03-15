import os
import json

fl = '/home/saksham/tts_experiments/dss_completions.jsonl'

with open(fl) as file:
    data = file.read()

print(data[0:10000])
