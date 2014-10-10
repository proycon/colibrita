#!/usr/bin/env python3

import os
import glob

header = "Configuration,Accuracy,Word Accuracy,Recall,BLEU,METEOR,NIST,TER,WER,PER"

for filename in glob.glob("ep7os12-*.summary.score"):
    name = '.'.join(os.path.basename(filename).split('.')[:-3])
    name = '-'.join(name.split('-')[1:])
    with open(filename) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            fields = [ float(x) for x in line.split(' ') ]
            fields = [ str(round(x,4)) for x in fields ]
            print(name + ',' + ','.join(fields))





