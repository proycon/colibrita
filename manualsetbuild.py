#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from colibrita.format import Reader, Writer

def main():
    if len(sys.argv) < 2:
        print("Syntax: set",file=sys.stderr)
        sys.exit(2)
    setfile= sys.argv[1:]

    sentencepairs = []
    if os.path.exists(setfile):
        reader = Reader(setfile)
        for sentencepair in reader:
            sentencepairs.append(sentencepair)



if __name__ == '__main__':
    main()
