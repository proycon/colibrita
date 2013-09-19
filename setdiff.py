#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import
import sys
from colibrita.format import Reader

def main():
    if len(sys.argv) == 0:
        print("Specify sets as arguments",file=sys.stderr)
        sys.exit(2)

    readers = []
    for filename in sys.argv[1:]:
        reader = Reader(filename)
        readers.append( (reader, iter(reader)) )

    while True:
        try:
            sentencepairs = []
            for i, (reader, readeriter) in enumerate(readers):
                sentencepair = next(readeriter)
                if i == 0:
                    pivotpair = sentencepair
                else:
                    if pivotpair.input != sentencepair.input:
                        print("Input sentences are not all equal.", file=sys.stderr)
                        sys.exit(2)
                sentencepairs.append( sentencepair )

            differences = 0
            for i, sentencepair in enumerate(sentencepairs):
                for j, sentencepair2 in enumerate(sentencepairs):
                    if i < j:
                        if sentencepair.outputstr() != sentencepair2.outputstr():
                            differences += 1

            if differences:
                print("Input: " + sentencepair.inputstr(True))
                print("Differences: ", differences)
                for i, sentencepair in enumerate(sentencepairs):
                    print(reader[i].filename + ": " + sentencepair.outputstr(True))
                print("-----------------------------------------------------------")


        except StopIteration:
            break


    for reader in readers:
        reader.close()

if __name__ == '__main__':
    main()
