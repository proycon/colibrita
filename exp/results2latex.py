#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import os
import glob
import datetime



header = ["System","Accuracy","Word Accuracy","Recall","BLEU","METEOR","NIST","TER","WER","PER"]
includefields = (0,1,2,3,4)


def printdata(data):
    print(r"\begin{tabular}{|l|" + "r" * (len(header) - 1) + "|}")
    print("%generated at " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + " in " + os.getcwd() )
    print(r"\hline")
    print(" & ".join(header) + r"\\")
    print(r"\hline")
    for key in sorted(data):
        if key.find('baseline') != -1:
            printrow(data,key)
    print(r"\hline")
    for key in sorted(data):
        if key.find('baseline') == -1:
            printrow(data, key)
    print(r"\hline")
    print(r"\end{tabular}")

def printrow(data, key):
    for i, field in enumerate(data[key]):
        if i in includefields:
            if i >= 1 and i <= 6 and i != 3:
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


data1 = {}
for key in data:
    if key.find('moses') == -1:
        data1[key] = data[key]
printdata(data1)


print()


data2 = {}
for key in data:
    if key.find('moses') != -1:
        data2[key] = data[key]
printdata(data2)


