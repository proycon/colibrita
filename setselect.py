#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from colibrita.format import Reader, Writer


def processbuffer(buffer, reader, writer, inputs, num):
    repeat = True
    while repeat:
        for i in range(0,10):
            print()

        print("===================== OFFSET " + str(num) + " ======================")
        for i, sentencepair in enumerate(buffer):
            print("----------------- #" + str(i+1) + " ------------------------")
            print(sentencepair.inputstr(True,"blue"))
            print(sentencepair.refstr(True,"green"))

        print("\n---------------------------------------------------------")
        print("Select any sentence pairs? Type space-separated list of numbers, q to quit:")
        selection = sys.stdin.readline().strip()
        if selection.lower() == 'q':
            return buffer, True
        try:
            selection = [ int(x) for x in selection.split() ]
        except ValueError:
            print("Invalid selection, try again...",file=sys.stderr)
            continue

        repeat = False
        for s in selection:
            try:
                sentencepair = buffer[s-1]
                writer.write(sentencepair)
                inputs.add( hash(sentencepair.input) )
            except IndexError:
                print("Invalid index: " + str(s) + ", IGNORING!" ,file=sys.stderr)
        buffer = []

    return buffer, False

def main():
    if len(sys.argv) != 3:
        print("Syntax: inputset outputset offset",file=sys.stderr)
        sys.exit(2)
    try:
        inputset, outputset, offset = sys.argv[1:]
        offset = int(offset)
    except:
        inputset, outputset = sys.argv[1:]
        offset = 1

    buffer = []
    BUFFERSIZE = 10
    tmpfile=False
    inputs = set()
    if os.path.exists(outputset):
        writer = Writer(outputset + '.tmp')
        reader = Reader(outputset)
        for sentencepair in reader:
            inputs.add( hash(sentencepair.input) )
            writer.write(sentencepair)
        tmpfile=True
    else:
        writer = Writer(outputset)

    num = 0
    reader = Reader(inputset)
    quit = False
    for sentencepair in reader:
        num += 1
        if not hash(sentencepair.input) in inputs:
            if num >= offset:
                buffer.append(sentencepair)
                if len(buffer) == BUFFERSIZE:
                    buffer, quit = processbuffer(buffer, reader,writer, inputs,num)
                    if quit: break

    if buffer and not quit: processbuffer(buffer, reader,writer, inputs,num)


    writer.close()
    if tmpfile:
        os.rename(outputset+'.tmp',outputset)

if __name__ == '__main__':
    main()
