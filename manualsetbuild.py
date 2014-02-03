#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from colibrita.format import Reader, Writer

def main():
    if len(sys.argv) < 4:
        print("Syntax: set L1 L2",file=sys.stderr)
        sys.exit(2)
    setfile= sys.argv[1]
    l1= sys.argv[2]
    l2= sys.argv[3]

    sentencepairs = []
    if os.path.exists(setfile):
        print("Loading existing file: ", setfile,file=sys.stderr)
        reader = Reader(setfile)
        for sentencepair in reader:
            sentencepairs.append(sentencepair)
        print(len(sentencepairs) + " sentences loaded",file=sys.stderr)
    else:
        print("New file: ", setfile,file=sys.stderr)

    print("Type h for help",file=sys.stderr)


    cursor = None

    quit = False
    while not quit:
        print(">" , end="" ,file=sys.stderr)

        cmd = sys.stdin.readline().strip()
        if cmd.lower() == 'q':
            quit = True
        elif cmd.lower() == 'h':
            print("q\tSave and quit",file=sys.stderr)
            print("n\tNew sentence pair",file=sys.stderr)
            print("d\tDelete sentence pair",file=sys.stderr)
            print("a\tAdd alternative",file=sys.stderr)
            print(">\tNext sentence pair",file=sys.stderr)
            print("<\tPrevious sentence pair",file=sys.stderr)
            print("12\tGo to sentence pair #12", file=sys.stderr)
        elif cmd.lower() == "<":
            if not cursor:
                cursor = len(sentencepairs) - 1
            else:
                cursor = cursor - 1
                if cursor < 0:
                    cursor = len(sentencepairs) - 1
            showsentencepair(sentencepairs, cursor)
        elif cmd.lower() == ">":
            if not cursor:
                cursor = 0
            else:
                cursor = cursor + 1
                if cursor >= len(sentencepairs):
                    cursor = 0
            showsentencepair(sentencepairs, cursor)
        elif cmd.lower().isdigit():
            cursor = int(cmd.lower()) - 1
            if cursor < 0:
                cursor = 0
            if cursor >= len(sentencepairs):
                cursor = len(sentencepairs) - 1
        elif cmd.lower() == 'n':
            cursor = newsentencepair(sentencepairs)

def showsentencepair(sentencepairs, cursor):
    sentencepair = sentencepairs[cursor]
    print("------------------ #" + str(cursor+1) + " of " + str(len(sentencepairs)) + "----------------",file=sys.stderr)
    print("Input:     " + sentencepair.inputstr(True,"blue"))
    print("Reference: " + sentencepair.refstr(True,"green"))
    if sentencepair.alternatives:
        for alt in sentencepair.alternatives:
            print("Alternative: " + str(alt))

def newsentencepair():
    cursor = len(sentencepairs)
    print("------------------ #" + str(cursor+1) + ": New sentence pair ----------------",file=sys.stderr)
    print("Enter untokenised text, mark L1 fragment in *asterisks*",file=sys.stderr)
    print("Input: " , end="" ,file=sys.stderr)
    input = sys.stdin.readline().strip()
    print("Reference: " , end="" ,file=sys.stderr)
    ref = sys.stdin.readline().strip()
    return cursor


if __name__ == '__main__':
    main()
