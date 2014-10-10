#!/usr/bin/env python3

import os
import glob

header = "Configuration,Accuracy,Word Accuracy,Recall,BLEU,METEOR,NIST,TER,WER,PER"

for filename in glob.glob("ep7os12-*.summary.score"):
    name = os.path.basename('-'.join(filename.split('.')[0].split('-')[1:]))
    with open(filename) as f:
        line = f.read() #header
        line = f.read()
        fields = [ str(round(float(x),4) for x in line.split(' ')) ]
        print(name + ',' + ','.join(fields))





