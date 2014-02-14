#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from colibrita.format import Reader, Writer


def main():
    try:
        outputset = sys.argv[1]
        inputsets = sys.argv[2:]
    except:
        print("Syntax: outputset inputset inputset2...",file=sys.stderr)
        sys.exit(2)

    id = 0
    writer = Writer(outputset)
    for inputset in inputsets:
        reader = Reader(inputset)
        for sentencepair in reader:
            id += 1
            sentencepair.id = id
            writer.write(sentencepair)
        reader.close()
    writer.close()

if __name__ == '__main__':
    main()
