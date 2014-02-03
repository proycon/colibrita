#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from collections import defaultdict
from colibrita.format import Reader, Writer

sources = defaultdict(int)
categories = defaultdict(int)

def main():
    global sources, categories
    if len(sys.argv) < 4:
        print("Syntax: set L1 L2",file=sys.stderr)
        sys.exit(2)
    setfile= sys.argv[1]
    l1= sys.argv[2]
    l2= sys.argv[3]

    sentencepairs = []
    if os.path.exists(setfile):
        print("Loading existing file: ", setfile)
        reader = Reader(setfile)
        for sentencepair in reader:
            sentencepairs.append(sentencepair)
            if sentencepair.source:
                sources[sentencepair.source] += 1
            if sentencepair.category:
                categories[sentencepair.category] += 1
        print(len(sentencepairs) + " sentences loaded")
    else:
        print("New file: ", setfile,file=sys.stderr)

    print("Type h for help")


    cursor = None

    quit = False
    while not quit:
        sys.stdout.write("> ")
        sys.stdout.flush()

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
            print("w\tWrite changes to disk", file=sys.stderr)
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
        elif cmd.lower() == 'w':
            writer = Writer(setfile)
            for sentencepair in sentencepairs:
                writer.write(sentencepair)
            writer.close()


def showsentencepair(sentencepairs, cursor):
    sentencepair = sentencepairs[cursor]
    print("------------------ #" + str(cursor+1) + " of " + str(len(sentencepairs)) + "----------------",file=sys.stderr)
    print("Input:     " + sentencepair.inputstr(True,"blue"))
    print("Reference: " + sentencepair.refstr(True,"green"))
    if sentencepair.alternatives:
        for alt in sentencepair.alternatives:
            print("Alternative: " + str(alt))
    if sentencepair.source:
        print("Source: " + sentencepair.source)
    if sentencepair.category:
        print("Category: " + sentencepair.category)


def newsentencepair(sentencepairs):
    global sources, categories
    cursor = len(sentencepairs)
    print("------------------ #" + str(cursor+1) + ": New sentence pair ----------------",file=sys.stderr)
    print("Enter untokenised text, mark L1 fragment in *asterisks*",file=sys.stderr)
    print("Input: " , end="" ,file=sys.stderr)
    input = sys.stdin.readline().strip()
    print("Reference: " , end="" ,file=sys.stderr)
    ref = sys.stdin.readline().strip()
    choices = listsources()
    print("Source: " , end="" ,file=sys.stderr)
    src = sys.stdin.readline().strip()
    if src.isdigit():
        if int(src) in choices:
            src = choices[int(src)]
        else:
            print("Invalid source, leaving empty",file=sys.stderr)
            src = None
    if src:
        sources[src] += 1
    choices = listcats()
    print("Category: " , end="" ,file=sys.stderr)
    cat = sys.stdin.readline().strip()
    if cat.isdigit():
        if int(cat) in choices:
            cat = choices[int(cat)]
        else:
            print("Invalid category, leaving empty",file=sys.stderr)
            cat = None
    return cursor

def listsources():
    choices = {}
    for i, (source,_) in enumerate(sorted(sources.items(), key=lambda x: -1 * x[1])):
        choices[i+1] = source
        print(" " + str(i+1) + ") " + source)
    return choices

def listcats():
    choices = {}
    for i, (cat,_) in enumerate(sorted(categories.items(), key=lambda x: -1 * x[1])):
        choices[i+1] = cat
        print(" " + str(i+1) + ") " + cat)
    return choices




if __name__ == '__main__':
    main()
