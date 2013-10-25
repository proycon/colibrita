#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from colibrita.format import Reader, Writer


def processbuffer(buffer, reader, writer, inputs):
    repeat = True
    while repeat:
        for i in range(0,10):
            print()

        for i, sentencepair in enumerate(buffer):
            print("----------------- #" + str(i+1) + " ------------------------")
            print(sentencepair.inputstr(True,"blue"))
            print(sentencepair.refstr(True,"green"))

        print("\n---------------------------------------------------------")
        print("Select any sentence pairs? Type space-separated list of numbers, q to quit:")
        selection = sys.stdin.readline().strip()
        if selection.lower() == 'q':
            return True
        try:
            selection = [ int(x) for x in selection.split() ]
        except ValueError:
            print("Invalid selection, try again...",file=sys.stderr)
            continue

        repeat = False
        buffer = []
        for s in selection:
            try:
                sentencepair = buffer[s-1]
            except IndexError:
                print("Invalid index: " + str(s) + ", IGNORING!" ,file=sys.stderr)
                writer.write(sentencepair)
                inputs.add( hash(sentencepair.input) )

    return False

def main():
    if len(sys.argv) != 3:
        print("Syntax: inputset outputset",file=sys.stderr)
        sys.exit(2)
    inputset, outputset = sys.argv[1:]

    buffer = []
    BUFFERSIZE = 10
    tmpfile=False
    inputs = {}
    if os.path.exists(outputset):
        writer = Writer(outputset + '.tmp')
        reader = Reader(outputset)
        for sentencepair in reader:
            inputs.add( hash(sentencepair.input) )
            writer.write(sentencepair)
        tmpfile=True
    else:
        writer = Writer(outputset)

    reader = Reader(inputset)
    quit = False
    for sentencepair in reader:
        if not hash(sentencepair.input) in inputs:
            buffer.append(sentencepair)
            if len(buffer) == BUFFERSIZE:
                quit = processbuffer(buffer, reader,writer, inputs)
                if quit: break

    if buffer and not quit: processbuffer(buffer, reader,writer, inputs)


    writer.close()
    if tmpfile:
        os.rename(outputset+'.tmp',outputset)

if __name__ == '__main__':
    main()
