#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import

import sys
from colibrita.format import Reader, Writer


def main():
    try:
        inputset = sys.argv[1]
        outputset = sys.argv[2]
    except:
        print("Syntax: inputset outputset",file=sys.stderr)
        sys.exit(2)

    reader = Reader(inputset)
    writer = Writer(outputset)
    for sentencepair in reader:
        sentencepair.ref = None
        sentencepair.source = None
        sentencepair.category = None
        writer.write(sentencepair)
    writer.close()
    reader.close()

if __name__ == '__main__':
    main()
