#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import os
import glob
import datetime


header = ["System","Accuracy","Word Accuracy","Recall","BLEU","METEOR","NIST","TER","WER","PER"]


def printdata(key):
    for i, field in enumerate(data[key]):
        if i >= 1 and i <= 6:
            highlight = True
            for k in data:
                if k != key:
                    if data[key][i] < data[k][i]:
                        highlight = False
                        break
        elif i > 6:
            highlight = True
            for k in data:
                if k != key:
                    if data[key][i] > data[k][i]:
                        highlight = False
                        break
        else:
            highlight = False

        if highlight:
            print('\\textbf{' + str(field) + '}', end='')
        else:
            print(str(field), end='')
        if i < len(data[key]) - 1:
            print(' & ',end='')
        else:
            print(r' \\')


data = {}
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
            fields = [ round(x,4) for x in fields ]
            data[name] = (name,) + tuple(fields)

print(r"\begin{tabular}{l|rrrrrrrrr}")
print("%generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + " in " + os.getcwd() )
print(r"\hline")
print(" & ".join(header) + r"\\")
print(r"\hline")
if 'baseline' in data: print(data['baseline'])
if 'lmbaseline' in data: print(data['lmbaseline'])
print(r"\hline")
for key in sorted(data):
    if key.find('baseline') == -1 and key.find('moses') == -1:
        printdata(key)
print(r"\hline")
print(r"\end{tabular}")

print()

print(r"\begin{tabular}{l|rrrrrrrrr}")
print("%generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + " in " + os.getcwd() )
print(r"\hline")
print(" & ".join(header) + r"\\")
print(r"\hline")
if 'mosesbaseline' in data: print(data['mosesbaseline'])
print(r"\hline")
for key in sorted(data):
    if key.find('baseline') == -1 and key.find('moses') != -1:
        printdata(key)
print(r"\hline")

print(r"\end{tabular}")
