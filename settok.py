#!/usr/bin/env python3
from __future__ import print_function, unicode_literals, division, absolute_import

import sys
import os
from colibrita.format import Reader, Writer, Fragment, Alternative


def tok(s, lang):
    if isinstance(s, tuple):
        s = " ".join(s)
    with open('/tmp/ucto.in','w',encoding='utf-8') as f:
        f.write(s)
    cmd = "ucto -L" + lang + " -mnS /tmp/ucto.in /tmp/ucto.out"
    r = os.system(cmd)
    if r != 0:
        raise Exception("Error during tokenisation: " + cmd)
    with open('/tmp/ucto.out','r',encoding='utf-8') as f:
        line = f.read()
    return tuple(line.strip().split(' '))


def main():
    try:
        inputset = sys.argv[1]
        outputset = sys.argv[1]
    except:
        print("Syntax: inputset outputset",file=sys.stderr)
        sys.exit(2)

    writer = Writer(outputset)
    reader = Reader(inputset)
    for sentencepair in reader:
        if sentencepair.ref:
            left, fragment, right = sentencepair.fragments(sentencepair.ref)
            if left.strip():
                left = tok(left, reader.L2)
            else:
                left = ""
            alts = fragment.alternatives
            fragment = Fragment(tok(fragment.value),reader.L2, id=1)
            for alt in alts:
                fragment.alternatives.append(Alternative(tok(alt)))
            if right.strip():
                right = tok(right, reader.L2)
            else:
                right = ""
            if left and right:
                ref = (left, fragment, right)
            elif left:
                ref = (left, fragment)
            elif right:
                ref = (fragment, right)
            sentencepair.ref = ref

        if sentencepair.out:
            left, fragment, right =sentencepair.fragments(sentencepair.output)
            if left.strip():
                left = tok(left, reader.L2)
            else:
                left = ""
            alts = fragment.alternatives
            fragment = Fragment(tok(fragment.value),reader.L2, id=1)
            for alt in alts:
                fragment.alternatives.append(Alternative(tok(alt)))
            if right.strip():
                right = tok(right, reader.L2)
            else:
                right = ""
            if left and right:
                out = (left, fragment, right)
            elif left:
                out = (left, fragment)
            elif right:
                out = (fragment, right)
            sentencepair.output = out

        if sentencepair.input:
            left, fragment, right =sentencepair.fragments(sentencepair.input)
            if left.strip():
                left = tok(left, reader.L2)
            else:
                left = ""
            alts = fragment.alternatives
            fragment = Fragment(tok(fragment.value),reader.L1, id=1)
            if right.strip():
                right = tok(right, reader.L2)
            else:
                right = ""
            if left and right:
                inp = (left, fragment, right)
            elif left:
                inp = (left, fragment)
            elif right:
                inp = (fragment, right)
            sentencepair.input = inp

        writer.write(sentencepair)
    reader.close()
    writer.close()

if __name__ == '__main__':
    main()
