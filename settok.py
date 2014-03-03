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
        outputset = sys.argv[2]
        l1 = sys.argv[3]
        l2 = sys.argv[4]
    except:
        print("Syntax: inputset outputset l1 l2",file=sys.stderr)
        sys.exit(2)

    writer = Writer(outputset)
    reader = Reader(inputset)
    for sentencepair in reader:
        if sentencepair.ref:
            for left, fragment, right in sentencepair.fragments(sentencepair.ref):
                print("Tokenising reference: L=", left,file=sys.stderr)
                print("                      F=", fragment.value,file=sys.stderr)
                print("                      R=", right,file=sys.stderr)
                if left.strip():
                    left = tok(left, l2)
                else:
                    left = ""
                alts = fragment.alternatives
                fragment = Fragment(tok(fragment.value,l2),id=fragment.id)
                for alt in alts:
                    fragment.alternatives.append(Alternative(tok(alt.value,l2)))
                if right.strip():
                    right = tok(right, l2)
                else:
                    right = ""
                if left and right:
                    ref = left + (fragment,) + right
                elif left:
                    ref = left + (fragment,)
                elif right:
                    ref = (fragment,) + right
                sentencepair.ref = ref

        if sentencepair.output:
            for left, fragment, right in sentencepair.fragments(sentencepair.output):
                print("Tokenising output:    L=", left,file=sys.stderr)
                print("                      F=", fragment.value,file=sys.stderr)
                print("                      R=", right,file=sys.stderr)
                if left.strip():
                    left = tok(left, l2)
                else:
                    left = ""
                alts = fragment.alternatives
                fragment = Fragment(tok(fragment.value,l2))
                for alt in alts:
                    fragment.alternatives.append(Alternative(tok(alt)))
                if right.strip():
                    right = tok(right, l2)
                else:
                    right = ""
                if left and right:
                    out = left + (fragment,) + right
                elif left:
                    out = left + (fragment,)
                elif right:
                    out = (fragment,) + right
                sentencepair.output = out

        if sentencepair.input:
            for left, fragment, right in sentencepair.fragments(sentencepair.input):
                print("Tokenising input:     L=", left,file=sys.stderr)
                print("                      F=", fragment.value,file=sys.stderr)
                print("                      R=", right,file=sys.stderr)
                if left.strip():
                    left = tok(left, l2)
                else:
                    left = ""
                alts = fragment.alternatives
                fragment = Fragment(tok(fragment.value,l1), id=fragment.id)
                if right.strip():
                    right = tok(right, l2)
                else:
                    right = ""
                if left and right:
                    inp = left + (fragment,) + right
                elif left:
                    inp = left + (fragment,)
                elif right:
                    inp = (fragment,) + right
                sentencepair.input = inp

        writer.write(sentencepair)
    reader.close()
    writer.close()

if __name__ == '__main__':
    main()
