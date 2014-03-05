#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import random
from colibrita.format import Reader, Writer


def main():
    try:
        inputset = sys.argv[1]
        outputset = sys.argv[2]
    except:
        print("Syntax: inputset outputset",file=sys.stderr)
        sys.exit(2)

    reader = Reader(inputset)
    sentencepairs = []
    for sentencepair in reader:
        sentencepairs.append(sentencepair)
    reader.close()
    writer = Writer(outputset)
    random.shuffle(sentencepairs)
    for i, sentencepair in enumerate(sentencepairs):
        sentencepair.id = i+1
        writer.write(sentencepair)
    writer.close()

if __name__ == '__main__':
    main()
