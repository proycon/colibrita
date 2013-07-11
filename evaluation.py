#!/usr/bin/env python3

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import sys
import os
import subprocess

from colibrita.format import Reader
from colibrita.common import log, runcmd, red, green, blue, yellow, white

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--matrexdir',type=str, help="Path to Matrex evaluation scripts",action='store',default="")
    parser.add_argument('--ref',type=str,help='Reference file', action='store',required=True)
    parser.add_argument('--out',type=str,help='Output file', action='store',required=True)
    parser.add_argument('--debug','-d', help="Debug", action='store_true', default=False)
    parser.add_argument('--workdir','-w',type=str,help='Work directorry', action='store',default=".")
    parser.add_argument('-i',dest='casesensitive',help='Measure translation accuracy without regard for case',action='store_false',default=True)
    args = parser.parse_args()

    totalavgaccuracy, totalwordavgaccuracy, matrexsrcfile, matrextgtfile, matrexoutfile = evaluate(Reader(args.ref), Reader(args.out), args.matrexdir, args.workdir, args.casesensitive)

    if args.matrexdir:
        mtscore(args.matrexdir, matrexsrcfile, matrextgtfile, matrexoutfile, args.workdir)


def evaluate(ref, out, matrexdir, workdir, casesensitive=True):
    ref_it = iter(ref)
    out_it = iter(out)

    matches = 0
    misses = 0

    wordmatches = 0
    wordmisses = 0

    accuracies = []
    wordaccuracies = []

    if casesensitive:
        eq = lambda x,y: x == y
    else:
        eq = lambda x,y: " ".join(x).lower() == " ".join(y).lower()

    matrexsrcfile = out.filename.replace('.xml','') + '.matrex-src.xml'
    matrextgtfile = out.filename.replace('.xml','') + '.matrex-ref.xml'
    matrexoutfile = out.filename.replace('.xml','') + '.matrex-out.xml'



    matrexsrc = open(matrexsrcfile ,'w', encoding='utf-8')
    matrextgt = open(matrextgtfile ,'w', encoding='utf-8')
    matrexout = open(matrexoutfile ,'w', encoding='utf-8')

    for t,f in (('src',matrexsrc),('ref',matrextgt),('tst',matrexout)):
        f.write( "<" + t + "set setid=\"mteval\" srclang=\"src\" trglang=\"tgt\">\n")
        f.write("<DOC docid=\"colibrita\" sysid=\"colibrita\">\n")

    while True:
        try:
            ref_s = next(ref_it)
            out_s = next(out_it)
        except StopIteration:
            break

        if ref_s.id != out_s.id:
            raise Exception("Sentence ID mismatch in reference and output! " + str(ref_s.id) + " vs " + str(out_s.id))
        elif ref_s.input != out_s.input:
            raise Exception("Sentence input mismatch in reference and output! " , ref_s.input,  " vs " , out_s.input)
        elif not ref_s.ref:
            raise Exception("No reference for sentence " + str(ref_s.id))
        elif not out_s.output:
            raise Exception("No output for sentence " + str(out_s.id))

        matrexsrc.write("<seg id=\"" + str(ref_s.id) + "\">" + ref_s.inputstr() + "</seg>\n")
        matrextgt.write("<seg id=\"" + str(ref_s.id) + "\">" + ref_s.refstr() + "</seg>\n")
        matrexout.write("<seg id=\"" + str(out_s.id) + "\">" + out_s.outputstr() + "</seg>\n")



        outputfragments = out_s.outputfragmentsdict()
        reffragments = ref_s.reffragmentsdict()
        for inputfragment in ref_s.inputfragmentsdict().values():
            if not inputfragment.id in reffragments:
                raise Exception("No reference fragment found for fragment " + str(inputfragment.id) + " in sentence " + str(ref_s.id))

            if not inputfragment.id in outputfragments:
                print("WARNING: Input fragment " + str(inputfragment.id) + " in sentence " + str(ref_s.id) + " is not translated!", file=sys.stderr)
                misses += 1
            else:
                if eq(reffragments[inputfragment.id].value, outputfragments[inputfragment.id].value):
                    matches += 1
                    wordmatches += 1
                else:
                    misses += 1
                    if len(reffragments[inputfragment.id].value) > len(outputfragments[inputfragment.id].value):
                        partialmatch = False
                        for i in range(0, len(reffragments[inputfragment.id].value)):
                            if eq(reffragments[inputfragment.id].value[i:i+len(outputfragments[inputfragment.id].value)], outputfragments[inputfragment.id].value):
                                partialmatch = True
                                break
                        if partialmatch:
                            p = len(outputfragments[inputfragment.id].value) / len(reffragments[inputfragment.id].value)
                            wordmatches += p
                            wordmisses += 1 - p
                        else:
                            wordmisses += 1
                    elif len(reffragments[inputfragment.id].value) < len(outputfragments[inputfragment.id].value):
                        partialmatch = False
                        for i in range(0, len(outputfragments[inputfragment.id].value)):
                            if eq(outputfragments[inputfragment.id].value[i:i+len(reffragments[inputfragment.id].value)], reffragments[inputfragment.id].value):
                                partialmatch = True
                                break
                        if partialmatch:
                            p = len(reffragments[inputfragment.id].value) / len(outputfragments[inputfragment.id].value)
                            wordmatches += p
                            wordmisses += 1 - p
                        else:
                            wordmisses += 1
                    else:
                        wordmisses += 1



            accuracy = matches/(matches+misses)
            print("Accuracy for sentence " + str(ref_s.id) + " = " + str(accuracy))
            accuracies.append(accuracy)


            wordaccuracy = wordmatches/(wordmatches+wordmisses)
            print("Word accuracy for sentence " + str(ref_s.id) + " = " + str(wordaccuracy))
            wordaccuracies.append(wordaccuracy)

    if accuracies:
        totalavgaccuracy = sum(accuracies) / len(accuracies)
        print("Total average accuracy = " + str(totalavgaccuracy))
    if wordaccuracies:
        totalwordavgaccuracy = sum(wordaccuracies) / len(wordaccuracies)
        print("Total word average accuracy = " + str(totalwordavgaccuracy))


    for t,f in (('src',matrexsrc),('ref',matrextgt),('tst',matrexout)):
        f.write("</DOC>\n</" + t + "set>")
        f.close()

    return totalavgaccuracy, totalwordavgaccuracy, matrexsrcfile, matrextgtfile, matrexoutfile


def mtscore(matrexdir, sourcexml, refxml, targetxml, WORKDIR = '.'):

    per = 0
    wer = 0
    bleu = 0
    meteor = 0
    nist = 0
    ter = 0

    EXEC_MATREX_WER = matrexdir + '/eval/WER_v01.pl'
    EXEC_MATREX_PER = matrexdir + '/eval/PER_v01.pl'
    EXEC_MATREX_BLEU = matrexdir + '/eval/bleu-1.04.pl'
    EXEC_MATREX_METEOR = matrexdir + '/meteor-0.6/meteor.pl'
    EXEC_MATREX_MTEVAL = matrexdir + '/mteval-v11b.pl'
    EXEC_MATREX_TER = matrexdir + '/tercom.jar'
    EXEC_PERL = 'perl'
    EXEC_JAVA = 'java'

    errors = False
    if EXEC_MATREX_BLEU and os.path.exists(EXEC_MATREX_BLEU):
        if not runcmd(EXEC_PERL + ' ' + EXEC_MATREX_BLEU + " -r " + refxml + ' -t ' + targetxml + ' -s ' + sourcexml + ' -ci > ' + 'bleu.score',  'Computing BLEU score'): errors = True
        if not errors:
            try:
                f = open(WORKDIR + '/bleu.score')
                for line in f:
                    if line[0:9] == "BLEUr1n4,":
                        bleu = float(line[10:].strip())
                        print("BLEU score: ", bleu, file=sys.stderr)
                f.close()
            except Exception as e:
                log("Error reading bleu.score:" + str(e),red)
                errors = True
    else:
        log("Skipping BLEU (no script found ["+EXEC_MATREX_BLEU+"])",yellow)

    if EXEC_MATREX_WER and os.path.exists(EXEC_MATREX_WER):
        if not runcmd(EXEC_PERL + ' ' + EXEC_MATREX_WER + " -r " + refxml + ' -t ' + targetxml + ' -s ' + sourcexml + '  > ' + 'wer.score', 'Computing WER score'): errors = True
        if not errors:
            try:
                f = open(WORKDIR + '/wer.score')
                for line in f:
                    if line[0:11] == "WER score =":
                        wer = float(line[12:20].strip())
                        log("WER score: " + str(wer), white)
                f.close()
            except Exception as e:
                log("Error reading wer.score:" + str(e),red)
                errors = True
    else:
        log("Skipping WER (no script found ["+EXEC_MATREX_WER+"]) ",yellow)

    if EXEC_MATREX_PER and os.path.exists(EXEC_MATREX_PER):
        if not runcmd(EXEC_PERL + ' ' + EXEC_MATREX_PER + " -r " + refxml + ' -t ' + targetxml + ' -s ' + sourcexml + '  > ' + 'per.score',  'Computing PER score'): errors = True
        if not errors:
            try:
                f = open(WORKDIR + '/per.score')
                for line in f:
                    if line[0:11] == "PER score =":
                        per = float(line[12:20].strip())
                        log("PER score: " + str(per), white)
                f.close()
            except Exception as e:
                log("Error reading per.score" + str(e),red)
                errors = True
    else:
        log("Skipping PER (no script found ["+EXEC_MATREX_PER+"])",yellow)

    if EXEC_MATREX_METEOR and os.path.exists(EXEC_MATREX_METEOR):
        if not runcmd(EXEC_PERL + ' -I ' + os.path.dirname(EXEC_MATREX_METEOR) + ' ' + EXEC_MATREX_METEOR + " -s colibrita -r " + refxml + ' -t ' + targetxml + ' --modules "exact"  > ' + 'meteor.score',  'Computing METEOR score'): errors = True
        if not errors:
            try:
                f = open(WORKDIR + '/meteor.score')
                for line in f:
                    if line[0:6] == "Score:":
                        meteor = float(line[7:].strip())
                        log("METEOR score: " + str(meteor), white)
                f.close()
            except Exception as e:
                log("Error reading meteor.score:" + str(e),red)
                errors = True
    else:
        log("Skipping METEOR (no script found ["+EXEC_MATREX_METEOR+"])",yellow)

    if EXEC_MATREX_MTEVAL and os.path.exists(EXEC_MATREX_MTEVAL):
        if not runcmd(EXEC_PERL + ' ' + EXEC_MATREX_MTEVAL + " -r " + refxml + ' -t ' + targetxml + ' -s ' + sourcexml +  '  > ' + 'mteval.score',  'Computing NIST & BLEU scores'): errors = True
        if not errors:
            try:
                f = open(WORKDIR + '/mteval.score')
                for line in f:
                    if line[0:12] == "NIST score =":
                        nist = float(line[13:21].strip())
                        log("NIST score: ", nist)
                    if line[21:33] == "BLEU score =":
                        try:
                            bleu2 = float(line[34:40].strip())
                            if bleu == 0:
                                bleu = bleu2
                                log("BLEU score: " + str(bleu), white)
                            elif abs(bleu - bleu2) > 0.01:
                                log("blue score from MTEVAL scripts differs too much: " + str(bleu) + " vs " + str(bleu2) +  ", choosing highest score")
                                if bleu2 > bleu:
                                    bleu = bleu2
                            else:
                                log("BLEU score (not stored): " + str(float(line[34:40].strip())))
                        except:
                            raise
                f.close()
            except Exception as e:
                log("Error reading mteval.score: " + str(e),red)
                errors = True
    else:
        log("Skipping MTEVAL (BLEU & NIST) (no script found)", yellow)

    if EXEC_MATREX_TER and os.path.exists(EXEC_MATREX_TER):
        if not runcmd(EXEC_JAVA + ' -jar ' + EXEC_MATREX_TER + " -r " + refxml + ' -h ' + targetxml + '  > ' + 'ter.score',  'Computing TER score'): errors = True
        if not errors:
            try:
                f = open(WORKDIR + '/ter.score')
                for line in f:
                    if line[0:10] == "Total TER:":
                        ter = float(line[11:].strip().split(' ')[0])
                        log("TER score: ", ter,white)
                f.close()
            except Exception as e:
                log("Error reading ter.score: " + str(e),red)
    else:
        log("Skipping TER (no script found)",yellow)


    log("SCORE SUMMARY\n===================\n")
    f = open(WORKDIR + '/summary.score','w')
    s = "BLEU METEOR NIST TER WER PER"
    f.write(s+ "\n")
    log(s)
    s = str(bleu) + " " + str(meteor) + " " + str(nist)  + " " + str(ter) + " " + str(wer)  + " " + str(per)
    f.write(s + "\n")
    log(s)
    f.close()


    return not errors




