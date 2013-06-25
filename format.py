from __future__ import print_function, unicode_literals, division, absolute_import

import lxml.etree
from lxml.builder import E
import sys

class Reader:
    def __init__(self, filename):
        self.stream = open(filename,'rb')

    def __iter__(self):
        parser = lxml.etree.iterparse(self.stream, events=("start","end"))
        for action, node in parser:
            if action == "end" and node.tag == "s":
                yield SentencePair.fromxml(node)

    def __del__(self):
        self.stream.close()

class Writer:
    def __init__(self, filename):
        self.stream = open(filename, 'wb')

    def write(self, sentencepair):
        assert isinstance(sentencepair, SentencePair)
        self.stream.write( lxml.etree.tostring(sentencepair.xml(), xml_declaration=False, pretty_print=True, encoding='utf-8') )

    def __del__(self):
        self.stream.close()


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
        for t in node.text.split():
            if t: content.append(t)
        for subnode in node:
            if subnode.tag == "f":
                content.append( Fragment(tuple([ x for x in subnode.text.split() if x ]), subnode.attrib.get('id',1) ) )
            else:
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

    def inputstr(self):
        return " ".join(self._str(self.input))

    def outputstr(self):
        return " ".join(self._str(self.output))

    def refstr(self):
        return " ".join(self._str(self.ref))

    def isref(self):
        return bool(self.ref)

    def isoutput(self):
        return bool(self.output)


    def _str(self, t):
        for x in t:
            if isinstance(x, Fragment):
                yield str(x)
            elif isinstance(x, str):
                yield " ".join(x)
            else:
                raise ValueError

    @staticmethod
    def _serialisevalue(v):
        result = []
        for x in v:
            if isinstance(x, Fragment):
                result.append(x.xml())
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
        assert isinstance(value, tuple)
        self.id = id
        self.value = value

    def __str__(self):
        return self.value

    def xml(self):
        return E.f(" ".join(self.value), id=str(self.id))

    def __eq__(self, other):
        return (self.id == other.id and self.value == other.value)
