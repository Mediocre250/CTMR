from collections import defaultdict
import os

import pandas as pd

path = '/home/djf/djf/POI/CDRF/data/Foursquare_NYC.txt'

dic = defaultdict(int)

f = open(path, 'r')
lines = f.readlines()

for line in lines:
    user, t, lat, lon, POI = line.strip().split('\t')
    dic[int(POI)] += 1



counts = [item[1] for item in dic.items()]

counts = sorted(counts)

with open('counts.txt', 'w') as f:
    for item in counts[:-10]:
        f.write('{}\n'.format(item))
