#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
import itertools
import base64
import timbl
from collections import defaultdict

from colibrita.format import Writer, Reader
from colibrita.common import extractpairs, makesentencepair, runcmd, makeset


class ClassifierExperts:
    def __init__(self, workdir):
        self.workdir = workdir
        self.classifiers = {}

    def load(self):
        #f = open(self.workdir + '/classifiers.cfg')
        #f.close()
        pass

    def counttranslations(self, reader):
        tcount = defaultdict(int)
        for sentencepair in reader:
            for left, sourcefragment, right in sentencepair.inputfragments():
                targetfragment = sentencepair.reffragments()[1]
                tcount[str(sourcefragment)][str(targetfragment)] += 1
        return tcount

    def countkeywords(self, reader, keywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold):
        print("Counting words for keyword extraction...", sys.stderr)
        wcount = defaultdict(int)
        wcount_total = 0
        kwcount = defaultdict(int)
        for sentencepair in reader:
            for left, sourcefragment, right in sentencepair.inputfragments():
                for word in sourcefragment:
                    wcount[word] += 1
                    wcount_total += 1
                for word in itertools.chain(left, right):
                    wcount[word] += 1
                    wcount_total += 1
                    targetfragment = sentencepair.reffragments()[1]
                    kwcount[str(sourcefragment)][str(targetfragment)][word] += 1


        return wcount, kwcount, wcount_total


    def probability_translation_given_keyword(self, source,target, keyword, kwcount, wcount):
        source = str(source)
        target = str(target)

        if not source in kwcount:
            print("focusword not seen", source, file=sys.stderr)
            return 0 #focus word has not been counted for

        if not target in kwcount[source]:
            print("target not seen:", target, file=sys.stderr)
            return 0 #sense has never been seen for this focus word

        if not keyword in wcount:
            print("keyword not seen:", keyword, file=sys.stderr)
            return 0 #keyword has never been seen

        Ns_kloc = 0.0
        if keyword in kwcount[source][target]:
            Ns_kloc = float(kwcount[source][target][keyword])

        Nkloc = 0
        for t in kwcount[source]:
            if keyword in kwcount[source][t]:
                Nkloc += kwcount[source][t][keyword]


        Nkcorp = wcount[keyword] #/ float(totalcount_sum)

        return (Ns_kloc / Nkloc) * (1/Nkcorp)


    def extract_keywords(self, sourcefragment, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, kwcount, wcount):
        print("Extracting keywords for " + sourcefragment + "...")

        sourcefragment = str(sourcefragment)

        if not sourcefragment in kwcount:
            print("WARNING: No count found!",file=sys.stderr)
            return [] #focus word has not been counted for


        bag = []
        #select all words that occur at least 3 times for a sense, and have a probability_sense_given_keyword >= 0.001
        for targetfragment in kwcount[sourcefragment]:
            for keyword, freq in kwcount[sourcefragment][targetfragment].items():
                 if (wcount[keyword] >= bow_filter_threshold): #filter very rare words (occuring less than 20 times)
                     if freq>= bow_absolute_threshold:
                        p = self.probability_translation_given_keyword(sourcefragment, targetfragment, keyword, kwcount, wcount)
                        if p >= bow_prob_threshold:
                            bag.append( (keyword, targetfragment, freq, p) )

        bag = sorted(bag)
        f = open(self.workdir + '/' + base64.b64encode(sourcefragment) + '.keywords','w',encoding='utf-8')
        for keyword, targetfragment, c, p in bag:
            f.write(keyword + '\t' + str(targetfragment) + '\t' + str(c) + '\t' + str(p) + '\n')
        f.close()

        print("\tFound " + str(len(bag)) + " keywords", file=sys.stderr)
        return bag





    def build(self, reader, leftcontext, rightcontext, dokeywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold, timbloptions):
        assert (isinstance(reader, Reader))



        if keywords:
            wcount, tcount, wcount_total = self.countkeywords(reader, keywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold)
        else:
            tcount = self.counttranslations(reader)
            wcount = {} #not needed

        reader.reset()

        keywords = {}

        #make translation table of direct translation that have only one translation
        dttable = open(self.workdir + '/directtranslation.table','w',encoding='utf-8')
        for source in tcount:
            for target in tcount[source]:
                if len(tcount[str(source)]) == 1:
                    dttable.write(str(source) + "\t" + str(target) + "\n")
            #gather keywords:
            if dokeywords:
                keywords[source] = self.extract_keywords(source, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, tcount, wcount)
        dttable.close()

        #now loop over corpus and build classifiers for those where disambiguation is needed
        for sentencepair in reader:
            for left, inputfragment, right in sentencepair.inputfragments():
                assert str(inputfragment) in tcount
                if len(tcount[str(inputfragment)]) > 1:
                    #extract local context
                    features = []
                    if leftcontext:
                        f_left = list(left[-leftcontext:])
                        if len(f_left) < leftcontext:
                            f_left = list(["<s>"] * (leftcontext - len(f_left))) + f_left
                    features += f_left

                    if rightcontext:
                        f_right = right[:rightcontext]
                        if len(f_right) < rightcontext:
                            f_right = f_right + list(["</s>"] * (rightcontext - len(f_right)))
                    features += f_right


                    targetfragment = sentencepair.reffragments()[1]

                    #extract global context
                    if dokeywords and str(inputfragment) in keywords:
                        bag = {}
                        for keyword, target, freq,p in keywords[str(inputfragment)]:
                            bag[keyword] = 0

                        for word in itertools.chain(left, right):
                            if word in bag:
                                bag[keyword] = 1

                        #add to features
                        for keyword in sorted(bag.keys()):
                            features.append(keyword+"="+str(bag[keyword]))

                    if not str(inputfragment) in self.classifiers:
                        #Build classifier
                        self.classifiers[str(inputfragment)] = timbl.TimblClassifier(self.workdir + '/' + base64.b64encode(str(inputfragment)), timbloptions)

                    self.classifiers[str(inputfragment)].append( features, str(targetfragment) )











def main():
    parser = argparse.ArgumentParser(description="Colibrita - Translation Assistance")
    parser.add_argument('--train',dest='settype', action='store_const',const='train')
    parser.add_argument('--test',dest='settype', action='store_const',const='test')
    parser.add_argument('-f','--dataset', type=str,help="Dataset file", action='store',required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-l','--leftcontext', help="Left local context size", action='store',default=0)
    parser.add_argument('-r','--rightcontext', help="Right local context size", action='store',default=0)
    parser.add_argument('-k','--keywords',help="Add global keywords in context", action='store_true',default=False)
    parser.add_argument("--kt",dest="bow_absolute_threshold", help="Keyword needs to occur at least this many times in the context (absolute number)", type=int, action='store',default=3)
    parser.add_argument("--kp",dest="bow_prob_threshold", help="minimal P(translation|keyword)", type=int, action='store',default=0.001)
    parser.add_argument("--kg",dest="bow_absolute_threshold", help="Keyword needs to occur at least this many times globally in the entire corpus (absolute number)", type=int, action='store',default=20)
    parser.add_argument("--ka",dest="compute_bow_params", help="Attempt to automatically compute --kt,--kp and --kg parameters", type=bool, action='store_false',default=True)
    parser.add_argument('-O', dest='timbloptions', help="Timbl Classifier options", type=str,action='store',default="-k 1")

    args = parser.parse_args()

    try:
        args.settype
    except:
        print("Specify either --train or --test")
        sys.exit(2)


    return True


if __name__ == '__main__':
    main()

