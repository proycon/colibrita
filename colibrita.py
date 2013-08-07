#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess
import itertools
import glob
import math
import timbl
from collections import defaultdict
from urllib.parse import quote_plus, unquote_plus
from copy import copy

from colibrita.format import Writer, Reader, Fragment
from colibrita.common import extractpairs, makesentencepair, runcmd, makeset, plaintext2sentencepair
from colibrita.baseline import makebaseline
from pynlpl.lm.lm import ARPALanguageModel
from pynlpl.formats.moses import PhraseTable

try:
    from twisted.web import server, resource
    from twisted.internet import reactor

    class ColibritaProcessorResource(resource.Resource):
        isLeaf = True
        numberRequests = 0

        def __init__(self, experts, dttable, ttable, lm, args):
            self.experts = experts
            self.dttable = dttable
            self.ttable = ttable
            self.lm = lm
            self.args = args

        def render_GET(self, request):
            self.numberRequests += 1
            if b'input' in request.args:
                request.setHeader(b"content-type", b"application/xml")
                line = str(request.args[b'input'],'utf-8')
                sentencepair = plaintext2sentencepair(line)
                if self.experts:
                    sentencepair = self.experts.processsentence(sentencepair, self.dttable, self.args.leftcontext, self.args.rightcontext, self.args.keywords, self.args.timbloptions + " +vdb -G0", self.lm, self.args.tmweight, self.args.lmweight)
                elif self.ttable:
                    pass #TODO
                return sentencepair.xml().encode('utf-8')
            else:
                request.setHeader(b"content-type", b"text/html")
                return b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<html>
  <head>
        <meta http-equiv="content-type" content="application/xhtml+xml; charset=utf-8"/>
        <title>Colibrita &cdot; Translation Assistant</title>
  </head>
  <body>
      Enter text in target language, enclose fall-back language content in asteriskes (*):<br />
      <form action="/" method="get">
          <input name="input" /><br />
          <input type="submit">
      </form>
  </body>
</html>"""

    class ColibritaServer:
        def __init__(self, port, experts, dttable, ttable, lm, args):
            assert isinstance(port, int)
            reactor.listenTCP(port, server.Site(ColibritaProcessorResource(experts,dttable, ttable,lm, args)))
            reactor.run()

except ImportError:
    print("(Webserver support not available)",file=sys.stderr)

MAXKEYWORDS = 100

class ClassifierExperts:
    def __init__(self, workdir):
        self.workdir = workdir
        self.classifiers = {}
        self.keywords = {}

    def load(self, timbloptions):
        for f in glob.glob(self.workdir + '/*.train'):
            sourcefragment = unquote_plus(os.path.basename(f).replace('.train',''))
            print("Loading classifier " + sourcefragment, file=sys.stderr)
            self.classifiers[sourcefragment] = timbl.TimblClassifier(f[:-6], timbloptions)
            conffile = f.replace('.train','.conf')
            if os.path.exists(conffile):
                configid, timblopts, accuracy = self.readconf(f[:-6])
                if timblopts: self.classifiers[sourcefragment].timbloptions += ' ' + timblopts
                print(" \- Loaded configuration " + configid, file=sys.stderr)
        print("Loaded " + str(len(self.classifiers)) + " classifiers",file=sys.stderr)
        self.loadkeywords()

    def loadkeywords(self):
        for f in glob.glob(self.workdir + '/*.keywords'):
            sourcefragment = unquote_plus(os.path.basename(f).replace('.keywords',''))
            self.keywords[sourcefragment] = []
            print("Loading keywords for " + sourcefragment, file=sys.stderr)
            f = open(f, 'r', encoding='utf-8')
            for line in f:
                keyword, target, c, p = line.split("\t")
                c = int(c)
                p = float(p)
                self.keywords[sourcefragment].append((keyword, target,c,p))
            f.close()


    def counttranslations(self, reader):
        tcount = defaultdict( lambda: defaultdict(int) )
        for sentencepair in reader:
            for left, sourcefragment, right in sentencepair.inputfragments():
                targetfragment = sentencepair.reffragmentsdict()[sourcefragment.id]
                tcount[str(sourcefragment)][str(targetfragment)] += 1
        return tcount

    def countkeywords(self, reader, keywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold):
        print("Counting words for keyword extraction...", file=sys.stderr)
        wcount = defaultdict(int)
        wcount_total = 0
        tcount = defaultdict( lambda: defaultdict(int) )
        kwcount = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for sentencepair in reader:
            if int(sentencepair.id) % 1000 == 0:
                print(" Counting words for keyword extraction @" + str(sentencepair.id), file=sys.stderr)
            for left, sourcefragment, right in sentencepair.inputfragments():
                for word in sourcefragment:
                    wcount[word] += 1
                    wcount_total += 1
                for word in itertools.chain(left.split(), right.split()):
                    wcount[word] += 1
                    wcount_total += 1
                    targetfragment = sentencepair.reffragmentsdict()[sourcefragment.id]
                    tcount[str(sourcefragment)][str(targetfragment)] += 1
                    kwcount[str(sourcefragment)][str(targetfragment)][word] += 1


        return wcount, tcount,kwcount, wcount_total


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
        print("Extracting keywords for " + sourcefragment + "...", file=sys.stderr)

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

        if bag:
            print("\tFound " + str(len(bag)) + " keywords: ", file=sys.stderr)
            bag = sorted(bag)
            f = open(self.workdir + '/' + quote_plus(sourcefragment) + '.keywords','w',encoding='utf-8')
            for keyword, targetfragment, c, p in bag:
                f.write(keyword + '\t' + str(targetfragment) + '\t' + str(c) + '\t' + str(p) + '\n')
                print("\t\t", keyword, file=sys.stderr)
            f.close()

        else:
            print("\tNo keywords found", file=sys.stderr)
        return bag





    def build(self, reader, leftcontext, rightcontext, dokeywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold, timbloptions):
        assert (isinstance(reader, Reader))


        if dokeywords:
            print("Counting keywords", file=sys.stderr)
            wcount, tcount, kwcount, wcount_total = self.countkeywords(reader, dokeywords, compute_bow_params, bow_absolute_threshold, bow_prob_threshold,bow_filter_threshold)
        else:
            print("Gathering initial occurrence count", file=sys.stderr)
            tcount = self.counttranslations(reader)
            wcount = {} #not needed
            kwcount = {} #not needed

        reader.reset()

        #make translation table of direct translation that have only one translation
        print("Writing direct translation table", file=sys.stderr)
        dttable = open(self.workdir + '/directtranslation.table','w',encoding='utf-8')
        for source in tcount:
            for target in tcount[source]:
                if len(tcount[str(source)]) == 1:
                    dttable.write(str(source) + "\t" + str(target) + "\t" + str(tcount[str(source)][str(target)]) + "\n")
            #gather keywords:
            if dokeywords:
                bag = self.extract_keywords(source, bow_absolute_threshold, bow_prob_threshold, bow_filter_threshold, kwcount, wcount)
                if bag:
                    self.keywords[source] = bag
        dttable.close()



        if not dokeywords and not leftcontext and not rightcontext:
            print("No classifiers needed, skipping...", file=sys.stderr)
            return

        index = open(self.workdir + '/index.table','w',encoding='utf-8')
        #now loop over corpus and build classifiers for those where disambiguation is needed
        for sentencepair in reader:
            targetfragments = sentencepair.reffragmentsdict()
            usedclassifier = False
            for left, inputfragment, right in sentencepair.inputfragments():
                if not str(inputfragment) in tcount:
                    print("WARNING: Inputfragment " + str(inputfragment) + " not found in count! Skipping!!", file=sys.stderr)
                    continue
                left = tuple(left.split())
                right = tuple(right.split())
                if len(tcount[str(inputfragment)]) > 1:
                    usedclassifier = True
                    #extract local context
                    features = []
                    f_left = []
                    if leftcontext:
                        f_left = list(left[-leftcontext:])
                        if len(f_left) < leftcontext:
                            f_left = list(["<s>"] * (leftcontext - len(f_left))) + f_left
                    features += f_left

                    f_right = []
                    if rightcontext:
                        f_right = list(right[:rightcontext])
                        if len(f_right) < rightcontext:
                            f_right = f_right + list(["</s>"] * (rightcontext - len(f_right)))
                    features += f_right

                    targetfragment = targetfragments[inputfragment.id]

                    #extract global context
                    if dokeywords and str(inputfragment) in self.keywords:
                        bag = {}
                        for keyword, target, freq,p in sorted(self.keywords[str(inputfragment)], key=lambda x: -1 *  x[3])[:MAXKEYWORDS]: #limit to 100 most potent keywords
                            bag[keyword] = 0

                        for word in itertools.chain(left, right):
                            if word in bag:
                                bag[keyword] = 1

                        #add to features
                        for keyword in sorted(bag.keys()):
                            features.append(keyword+"="+str(bag[keyword]))

                    if not str(inputfragment) in self.classifiers:
                        #Build classifier
                        cid = quote_plus(str(inputfragment))
                        cidfile = self.workdir + '/' +  cid
                        print("\tInitiating classifier for " + str(inputfragment) + " -- " + cidfile + '.train',file=sys.stderr )
                        self.classifiers[str(inputfragment)] = timbl.TimblClassifier(cidfile, timbloptions)
                        index.write(cid + "\t" + str(inputfragment) + "\t" + str(sum(tcount[str(inputfragment)].values())) + "\t")
                        for target, c in tcount[str(inputfragment)].items():
                            index.write(str(target) + "\t" + str(c))
                        index.write("\n")

                    if not features:
                        print("WARNING: No features extracted for " + str(inputfragment) + ", skipping classifier!", file=sys.stderr)
                    else:
                        print("\tAppending training instance: " +str(inputfragment) + " -> " + str(targetfragment),file=sys.stderr )
                        self.classifiers[str(inputfragment)].append( features, str(targetfragment) )

            if usedclassifier:
                print("Built classifier(s) for @" + str(sentencepair.id), file=sys.stderr)
            elif int(sentencepair.id) % 1000 == 0:
                print("Skipped @" + str(sentencepair.id), file=sys.stderr)
        index.close()



    def gettimblskipopts(self, classifier, leftcontext,rightcontext,newleftcontext,newrightcontext, skipkeywords):
        skip = []
        for i in range(1,leftcontext+rightcontext+1): #TODO: TEST!!
            if i <= leftcontext:
                if i <= leftcontext - newleftcontext:
                    skip.append(i)
            elif i - leftcontext <= rightcontext:
                if i - leftcontext > newrightcontext:
                    skip.append(i)

        if skipkeywords:
            return "-mO:I" + ",".join([ str(i) for i in skip ]) + "," + str(leftcontext+rightcontext+1) + "-999" #timbl allows out of range ends
        else:
            return "-mO:I" + ",".join([ str(i) for i in skip ])

    def leaveoneouttest(self, classifier, leftcontext, rightcontext, dokeywords,  newleftcontext, newrightcontext, newdokeywords, timbloptions):
        print("Auto-configuring " + str(len(self.classifiers)) + " classifiers, determining optimal feature configuration using leave-one-out", file=sys.stderr)
        assert newleftcontext <= leftcontext
        assert newrightcontext <= rightcontext
        timblskipopts = self.gettimblskipopts(classifier, leftcontext, rightcontext, newleftcontext, newrightcontext, dokeywords and not newdokeywords )

        #leave one out classifier
        c = timbl.TimblClassifier(self.classifiers[classifier].fileprefix, timbloptions + " " + timblskipopts + " -t leave_one_out")
        c.train()
        accuracy = c.test(self.classifiers[classifier].fileprefix + ".train")
        return accuracy, timblskipopts




    def autoconf(self, leftcontext, rightcontext, dokeywords, timbloptions):
        print("Auto-configuring " + str(len(self.classifiers)) + " classifiers, determining optimal feature configuration using leave-one-out", file=sys.stderr)
        best = 0
        bestconfig = None
        for classifier in self.classifiers:
            self.classifiers[classifier].flush()
            for c in range(1,max(leftcontext,rightcontext)):
                print("\tTesting '" + classifier + "' with configuration l" + str(c) + "r" + str(c), file=sys.stderr)
                accuracy, timblskipopts = self.leaveoneouttest(classifier, leftcontext, rightcontext, dokeywords,  c, c, False, timbloptions)
                if accuracy > best:
                    bestconfig = (c,c,False, timblskipopts)
                    best = accuracy
                if dokeywords:
                    accuracy, timblskopopts = self.leaveoneouttest(classifier, leftcontext, rightcontext, dokeywords, c, c, True, timbloptions)
                    if accuracy > best:
                        bestconfig = (c,c,False, timblskipopts)
                        best = accuracy

            configid = 'l' + str(bestconfig[0]) + 'r' + str(bestconfig[1])
            if bestconfig[2]: configid += 'k'

            f = open(self.classifiers[classifier].fileprefix + '.conf', 'w',encoding='utf-8')
            f.write("config=" + configid+"\n")
            f.write("timblopts=" + bestconfig[3] + "\n")
            f.write("accuracy=" + str(best) + "\n")
            f.close()
            print("\tBest configuration for '" + classifier + "' is " + configid , file=sys.stderr)


    def readconf(self, classifier):
        configid = ""
        timblopts = ""
        accuracy = 0.0
        f = open(self.classifiers[classifier].fileprefix + '.conf', 'r',encoding='utf-8')
        for line in f:
            line = line.strip()
            if line[0:7] == 'config=':
                configid = line[8:]
            elif line[0:10] == 'timblopts=':
                timblopts = line[11:]
            elif line[0:9] == 'accuracy=':
                accuracy = float(line[10:])
            elif line and line[0] != '#':
                raise ValueError("readconf(): Unable to parse: " + line)
        f.close()
        return configid, timblopts, accuracy

    def train(self):
        print("Training " + str(len(self.classifiers)) + " classifiers", file=sys.stderr)
        for classifier in self.classifiers:
            self.classifiers[classifier].flush()
            if os.path.exists(self.classifiers[classifier].fileprefix + '.conf'):
                configid, timblopts, accuracy = self.readconf(classifier)
                if timblopts: self.classifiers[classifier].timbloptions += ' ' + timblopts
                print("\tLoaded configuration " + configid + " for '" + classifier + "'", file=sys.stderr)
            if os.path.exists(self.classifiers[classifier].fileprefix + '.train'):
                print("\tTraining '" + classifier + "'", file=sys.stderr)
                self.classifiers[classifier].train()
                self.classifiers[classifier].save()




    def processsentence(self, sentencepair, dttable, leftcontext, rightcontext, dokeywords, timbloptions, lm=None,tweight=1,lmweight=1):
        print("Processing sentence " + str(sentencepair.id),file=sys.stderr)
        sentencepair.ref = None
        sentencepair.output = copy(sentencepair.input)
        for left, inputfragment, right in sentencepair.inputfragments():
            left = tuple(left.split())
            right = tuple(right.split())
            if str(inputfragment) in dttable:
                #direct translation
                outputfragment = Fragment(tuple(dttable[str(inputfragment)].split()), inputfragment.id)
                print("\tDirect translation " + str(inputfragment) + " -> " + str(outputfragment), file=sys.stderr)
            elif str(inputfragment) in self.classifiers:
                #translation by classifier
                features = []

                if leftcontext:
                    f_left = list(left[-leftcontext:])
                    if len(f_left) < leftcontext:
                        f_left = list(["<s>"] * (leftcontext - len(f_left))) + f_left
                    features += f_left

                if rightcontext:
                    f_right = list(right[:rightcontext])
                    if len(f_right) < rightcontext:
                        f_right = f_right + list(["</s>"] * (rightcontext - len(f_right)))
                    features += f_right


                #extract global context
                if dokeywords and str(inputfragment) in self.keywords:
                    bag = {}
                    for keyword, target, freq,p in sorted(self.keywords[str(inputfragment)], key=lambda x: -1 *  x[3])[:MAXKEYWORDS]: #limit to 100 most potent keywords
                        bag[keyword] = 0

                    for word in itertools.chain(left, right):
                        if word in bag:
                            bag[keyword] = 1

                    #add to features
                    for keyword in sorted(bag.keys()):
                        features.append(keyword+"="+str(bag[keyword]))

                #pass to classifier
                print("\tClassifying '" + str(inputfragment) + "' ...", file=sys.stderr)
                classlabel, distribution, distance =  self.classifiers[str(inputfragment)].classify(features)
                classlabel = classlabel.replace('\_',' ')
                if lm and len(distribution) > 1:
                    print("\tClassifier translation prior to LM: " + str(inputfragment) + " -> [ DISTRIBUTION:" + str(repr(distribution))+" ]", file=sys.stderr)
                    candidatesentences = []
                    bestlmscore = -999999999
                    besttscore = -999999999
                    for targetpattern, score in distribution.items():
                        assert score >= 0 and score <= 1
                        tscore = math.log(score) #convert to base-e log (LM is converted to base-e upon load)
                        translation = tuple(targetpattern.split())
                        outputfragment = Fragment(translation, inputfragment.id)
                        candidatesentence = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
                        lminput = " ".join(sentencepair._str(candidatesentence)).split(" ") #joining and splitting deliberately to ensure each word is one item
                        lmscore = lm.score(lminput)
                        assert lmscore <= 0
                        if lmscore > bestlmscore:
                            bestlmscore = lmscore
                        if tscore > besttscore:
                            besttscore = tscore
                        candidatesentences.append( ( candidatesentence, outputfragment, tscore, lmscore ) )

                    #get the strongest sentence
                    maxscore = -9999999999
                    for candidatesentence, targetpattern, tscore, lmscore in candidatesentences:
                        tscore = tweight * (tscore-besttscore)
                        lmscore = lmweight * (lmscore-bestlmscore)
                        score = tscore + lmscore
                        print("\t LM candidate " + str(inputfragment) + " -> " + str(targetpattern) + "   score=tscore+lmscore=" + str(tscore) + "+" + str(lmscore) + "=" + str(score), file=sys.stderr)
                        if score > maxscore:
                            maxscore = score
                            outputfragment = targetpattern  #Fragment(targetpattern, inputfragment.id)
                    print("\tClassifier translation after LM: " + str(inputfragment) + " -> " + str(outputfragment) + " score= " + str(score), file=sys.stderr)

                else:
                    outputfragment = Fragment(tuple(classlabel.split()), inputfragment.id)
                    print("\tClassifier translation " + str(inputfragment) + " -> " + str(outputfragment) + "\t[ DISTRIBUTION:" + str(repr(distribution))+" ]", file=sys.stderr)

            else:
                #no translation found
                outputfragment = Fragment(None, inputfragment.id)
                print("\tNo translation for " + str(inputfragment), file=sys.stderr)
            sentencepair.output = sentencepair.replacefragment(inputfragment, outputfragment, sentencepair.output)
        return sentencepair

    def loaddttable(self):
        return loaddttable(self.workdir + '/directtranslation.table')

    def test(self, data, outputfile, leftcontext, rightcontext, dokeywords, timbloptions, lm=None,tweight=1,lmweight=1):
        dttable = self.loaddttable()
        writer = Writer(outputfile)
        for sentencepair in data:
            sentencepair = self.processsentence(sentencepair, dttable, leftcontext, rightcontext, dokeywords, timbloptions, lm, tweight, lmweight)
            writer.write(sentencepair)
        writer.close()


def loaddttable(filename):
    dttable = {}
    f = open(filename)
    for line in f:
        if line:
            fields = line.split('\t')
            dttable[fields[0]] = fields[1]
    f.close()
    return dttable




def main():
    parser = argparse.ArgumentParser(description="Colibrita - Translation Assistance", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train',dest='settype',help="Training mode", action='store_const',const='train')
    parser.add_argument('--test',dest='settype',help="Test mode (against a specific test set)", action='store_const',const='test')
    parser.add_argument('--run',dest='settype',help="Run mode (reads input from stdin)", action='store_const',const='run')
    parser.add_argument('--server',dest='settype', help="Server mode (RESTFUL HTTP Server)", action='store_const',const='server')
    parser.add_argument('--igen',dest='settype',help="Instance generation without actual training", action='store_const',const='igen')
    parser.add_argument('-f','--dataset', type=str,help="Dataset file", action='store',default="",required=False)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('-a','--autoconf', help="Automatically determine best feature configuration per expert (validated using leave-one-out), values for -l and -r are considered maxima, set -k to consider keywords, only needs to be specified at training time", action='store_true',default=False)
    parser.add_argument('-l','--leftcontext',type=int, help="Left local context size", action='store',default=0)
    parser.add_argument('-r','--rightcontext',type=int,help="Right local context size", action='store',default=0)
    parser.add_argument('-k','--keywords',help="Add global keywords in context", action='store_true',default=False)
    parser.add_argument("--kt",dest="bow_absolute_threshold", help="Keyword needs to occur at least this many times in the context (absolute number)", type=int, action='store',default=3)
    parser.add_argument("--kp",dest="bow_prob_threshold", help="minimal P(translation|keyword)", type=int, action='store',default=0.001)
    parser.add_argument("--kg",dest="bow_filter_threshold", help="Keyword needs to occur at least this many times globally in the entire corpus (absolute number)", type=int, action='store',default=20)
    parser.add_argument("--ka",dest="compute_bow_params", help="Attempt to automatically compute --kt,--kp and --kg parameters", action='store_false',default=True)
    parser.add_argument('-O', dest='timbloptions', help="Timbl Classifier options", type=str,action='store',default="-k 1")
    parser.add_argument('-o','--output',type=str,help="Output prefix", required = True)
    parser.add_argument('--baseline', help="Baseline test (use with --test, requires no previous --train)", action='store_true',default=False)
    parser.add_argument('--lm',type=str, help="Use language model in testing (file in ARPA format, as produced by for instance SRILM)", action='store',default="")
    parser.add_argument('--lmweight',type=float, help="Language model weight (when --lm is used)", action='store',default=1)
    parser.add_argument('--tmweight',type=float, help="Translation model weight (when --lm is used)", action='store',default=1)
    parser.add_argument('--port',type=int, help="Server port (use with --server)", action='store',default=7893)
    parser.add_argument('-T','--ttable', type=str,help="Phrase translation table (file) to use when testing with --lm and without classifier training", action='store',default="")

    args = parser.parse_args()

    try:
        if not args.settype in ['train','test','run','server', 'igen']:
            raise ValueError
    except:
        print("Specify either --train, --test, --run, --server, --igen")
        sys.exit(2)




    if args.settype == 'train' or args.settype == 'igen':
        if args.baseline:
            print("Baseline does not need further training, use --test instead", file=sys.stderr)
            sys.exit(2)
        elif not args.dataset:
            print("Specify a dataset to use for training! (-f)", file=sys.stderr)
            sys.exit(2)
        elif args.lm:
            print("WARNING: Language model specified during training, will be ignored", file=sys.stderr)

        experts = ClassifierExperts(args.output)

        data = Reader(args.dataset)
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        if not os.path.exists(args.output + '/directtranslation.table'):
            print("Building classifiers", file=sys.stderr)
            experts.build(data, args.leftcontext, args.rightcontext, args.keywords, args.compute_bow_params, args.bow_absolute_threshold, args.bow_prob_threshold, args.bow_filter_threshold, args.timbloptions + " -vdb -G0")
        elif args.settype == 'train':
            print("Classifiers already built", file=sys.stderr)
            experts.load(args.timbloptions + " +vdb -G0")
        else:
            print("Instances already generated",file=sys.stderr)
        if args.settype == 'train' and args.autoconf:
            experts.autoconf(args.leftcontext, args.rightcontext, args.keywords, args.timbloptions + " -vdb -G0")
        if args.settype == 'train': experts.train()
    elif args.settype == 'test':

        if not args.dataset:
            print("Specify a dataset to use for testing! (-f)", file=sys.stderr)
            sys.exit(2)
        if args.autoconf:
            print("Warning: Autoconf specified at testing time, has no effect. Has to be specified at training time", file=sys.stderr)

        if args.lm:
            print("Loading Language model", file=sys.stderr)
            lm = ARPALanguageModel(args.lm)
        else:
            lm = None

        if args.leftcontext or args.rightcontext or args.keywords:
            if not os.path.isdir(args.output):
                print("Output directory " + args.output + " does not exist, did you forget to train the system first?", file=sys.stderr)
                sys.exit(2)
            if not os.path.exists(args.output + '/directtranslation.table'):
                print("Direct translation table does not exist, did you forget to train the system first?", file=sys.stderr)
                sys.exit(2)
            experts = ClassifierExperts(args.output)
            print("Loading classifiers",file=sys.stderr)
            experts.load(args.timbloptions + " +vdb -G0")
            print("Running...",file=sys.stderr)
            data = Reader(args.dataset)
            experts.test(data, args.output + '.output.xml', args.leftcontext, args.rightcontext, args.keywords, args.timbloptions + " +vdb -G0", lm, args.tmweight, args.lmweight)
        elif args.ttable:
            print("Loading translation table",file=sys.stderr)
            ttable = PhraseTable(args.ttable,False, False, "|||", 3, 0,None, None)
            data = Reader(args.dataset)
            print("Making baseline",file=sys.stderr)
            if args.lm:
                print("(with LM)",file=sys.stderr)
                makebaseline(ttable, args.output + '.output.xml', data, lm, args.tmweight, args.lmweight)
            elif args.baseline:
                makebaseline(ttable, args.output + '.output.xml', data)
        else:
            print("Don't know what to do! Specify some classifier options or -T with --lm or --baseline", file=sys.stderr)
    elif args.settype == 'run' or args.settype == 'server':

        if args.settype == 'server':
            try:
                ColibritaServer
            except:
                print("Server not available, twisted not loaded...", file=sys.stderr)
                sys.exit(2)

        if args.lm:
            print("Loading Language model", file=sys.stderr)
            lm = ARPALanguageModel(args.lm)
        else:
            lm = None

        if args.autoconf:
            print("Warning: Autoconf specified at testing time, has no effect. Has to be specified at training time", file=sys.stderr)

        experts = None
        dttable = {}
        ttable = None
        if args.leftcontext or args.rightcontext or args.keywords:
            if not os.path.exists(args.output + '/directtranslation.table'):
                print("Direct translation table does not exist, did you forget to train the system first?", file=sys.stderr)
                sys.exit(2)
            else:
                print("Loading direct translation table", file=sys.stderr)
                dttable = loaddttable(args.output + '/directtranslation.table')
            experts = ClassifierExperts(args.output)
            print("Loading classifiers",file=sys.stderr)
            experts.load(args.timbloptions + " +vdb -G0")
        elif args.ttable:
            print("Loading translation table",file=sys.stderr)
            ttable = PhraseTable(args.ttable,False, False, "|||", 3, 0,None, None)

        if args.settype == 'run':
            print("Reading from standard input, enclose words/phrases in fallback language in asteriskes (*), type q<enter> to quit",file=sys.stderr)
            for line in sys.stdin:
                line = line.strip()
                if line == "q":
                    break
                else:
                    sentencepair = plaintext2sentencepair(line)
                    if experts:
                        sentencepair = experts.processsentence(sentencepair, dttable, args.leftcontext, args.rightcontext, args.keywords, args.timbloptions + " +vdb -G0", lm, args.tmweight, args.lmweight)
                    elif args.ttable:
                        pass #TODO
                    print(sentencepair.outputstr())
        elif args.settype == 'server':
            print("Starting Colibrita server on port " + str(args.port),file=sys.stderr)
            ColibritaServer(args.port, experts, dttable, ttable, lm, args)

    print("All done.", file=sys.stderr)
    return True


if __name__ == '__main__':
    main()
    sys.exit(0)

