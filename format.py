from __future__ import print_function, unicode_literals, division, absolute_import

import lxml.etree
from lxml.builder import E
import sys

class Reader:
    def __init__(self, filename):
        self.filename = filename
        self.stream = open(filename,'rb')

    def __iter__(self):
        parser = lxml.etree.iterparse(self.stream, events=("start","end"))
        for action, node in parser:
            if action == "end" and node.tag == "s":
                yield SentencePair.fromxml(node)

    def close(self):
        self.stream.close()
        self.stream = None


    def reset(self):
        self.stream.seek(0)

    def __del__(self):
        if self.stream: self.stream.close()

class Writer:
    def __init__(self, filename):
        self.filename = filename
        self.stream = open(filename, 'w')
        self.stream.write('<sentencepairs>\n')

    def write(self, sentencepair):
        assert isinstance(sentencepair, SentencePair)
        self.stream.write( str(lxml.etree.tostring(sentencepair.xml(), xml_declaration=False, pretty_print=True, encoding='utf-8'),'utf-8') )

    def close(self):
        self.stream.write('</sentencepairs>\n')
        self.stream.close()
        self.stream = None


    def __del__(self):
        if self.stream: self.close()

class SentencePair:
    def __init__(self, id,input, output, ref=None):
        if not isinstance(input, tuple):
            raise ValueError("Input - Expected tuple, got " + str(type(input)), input)
        if not isinstance(output, tuple) and not (output is None):
            raise ValueError("Output - Expected tuple, got " + str(type(output)), output)
        if not isinstance(ref, tuple) and not (ref is None):
            raise ValueError("Ref - Expected tuple, got " + str(type(ref)), ref)
        self.id = id
        self.input = input
        self.output = output
        self.ref = ref


    @staticmethod
    def _parsevalue(node):
        content = []
        if node.text:
            for t in node.text.split():
                 if t: content.append(t)
        for subnode in node:
            if subnode.tag == "f":
                content.append( Fragment(tuple([ x for x in subnode.text.split() if x ]), subnode.attrib.get('id',1) ) )
            elif subnode.text:
                for t in subnode.text.split():
                    if t: content.append(t)
            if subnode.tail: content.append(subnode.tail)
        return tuple(content)


    @staticmethod
    def fromxml(node):
        input = ref = output = None
        for subnode in node:
            if subnode.tag == 'input':
                input = SentencePair._parsevalue(subnode)
            elif subnode.tag == 'ref':
                ref = SentencePair. _parsevalue(subnode)
            elif subnode.tag == 'output':
                output = SentencePair._parsevalue(subnode)
        return SentencePair(node.attrib.get('id',1), input,output,ref)

    def replacefragment(self, old,new,s):
        s2 = []
        if s:
            for x in s:
                if s == old:
                    s2.append(new)
                else:
                    s2.append(x)
        return tuple(s2)


    def fragments(self, s):
        d = {}
        if s:
            for x in s:
                if isinstance(x, Fragment):
                    left = ""
                    right = ""
                    mode = 0
                    for y in s:
                        if isinstance(y,Fragment):
                            if x == y:
                                mode = 1
                        else:
                            if mode == 0:
                                left += " " + y
                            else:
                                right += y + " "
                    d[x.id] = left.strip(), x, right.strip()
        return d

    def inputfragments(self):
        return self.fragments(self.input)


    def outputfragments(self):
        return self.fragments(self.output)


    def reffragments(self):
        return self.fragments(self.ref)

    def inputstr(self):
        return " ".join(SentencePair._str(self.input))

    def outputstr(self):
        return " ".join(SentencePair._str(self.output))

    def refstr(self):
        return " ".join(SentencePair._str(self.ref))

    def isref(self):
        return bool(self.ref)

    def isoutput(self):
        return bool(self.output)

    @staticmethod
    def _str(t):
        s = ""
        for x in t:
            if isinstance(x, Fragment):
                for y in x.value:
                    yield y
            elif isinstance(x, str):
                yield x
            else:
                raise ValueError

    @staticmethod
    def _serialisevalue(v):
        result = []
        l = len(v)
        for i, x in enumerate(v):
            if isinstance(x, Fragment):
                if i > 0: result.append(" ")
                result.append(x.xml())
                if i < l - 1: result.append(" ")
            elif result and isinstance(result[-1], str):
                result[-1] += " " + x
            else:
                result.append(x)
        return result


    def xml(self):
        children = []

        if self.input: children.append( E.input(*SentencePair._serialisevalue(self.input)))
        if self.output: children.append( E.output(*SentencePair._serialisevalue(self.output)))
        if self.ref: children.append( E.ref(*SentencePair._serialisevalue(self.ref)))
        return E.s(*children, id = str(self.id))

class Fragment:
    def __init__(self, value,id=1):
        assert isinstance(value, tuple) or value is None
        self.id = id
        self.value = value

    def __str__(self):
        if self.value:
            return " ".join(self.value)
        else:
            return "{?}"

    def __len__(self):
        if self.value:
            return len(self.value)
        else:
            return 0

    def __iter__(self):
        if self.value:
            for word in self.value:
                yield word

    def __hash__(self):
        if self.value:
            return hash(self.value)
        else:
            return 0

    def xml(self):
        if self.value:
            return E.f(" ".join(self.value), id=str(self.id))
        else:
            return E.f(id=str(self.id))

    def __eq__(self, other):
        return (self.id == other.id and self.value == other.value)
