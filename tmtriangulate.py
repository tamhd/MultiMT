#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ./tmtriangulate.py features_based -ps test/model1 -pt test/model2 -o test/phrase-table_sample -t tempdir
#  This class implement a naive method for triangulation: nothing
#  The most important part of this method is to initialize variables

#TODO: It's still triangulation but not naive anymore
#Compute the data based solely on the co-occurrences, but keep the probability in case of emergency
# ref: /a/merkur3/thoang/eman/ufal-smt-playground/multi_playground/s.mosesgiza.f282bc2e.20140906-1739/moses/scripts/training/LexicalTranslationModel.pm get_lexical_counts
#TODO: Cope with the problem of NULL pointer
#TODO: Cope with the problem of overloading memory
#TODO: Cope with the problem of overloading hard drive, stop writing the src phrase count and remove the used file
#TODO: Cope with the problem of time constrains, parallelism
#TODO: July 10: polishing the source
#TODO: July 10: another option to compute the new features, keeping the counts from source phrase table
#TODO: After July 10: Balancing size of phrase table

from __future__ import division, unicode_literals
import sys
import os
import gzip
import argparse
import copy
import re
from math import log, exp, sqrt
from collections import defaultdict
from operator import mul
from tempfile import NamedTemporaryFile
#from tmcombine import Moses, Moses_Alignment, to_list
from subprocess import Popen
from multiprocessing import Pool,Value,Process
from datetime import datetime

# set the locale
#import locale
#locale.setlocale(locale.LC_ALL, "C")

try:
    from itertools import izip
except:
    izip = zip

def parse_command_line():

    parser = argparse.ArgumentParser(description="Combine translation models. Check DOCSTRING of the class Triangulate_TMs() and its methods for a more in-depth documentation and additional configuration options not available through the command line. The function test() shows examples")

    group1 = parser.add_argument_group('Main options')
    group2 = parser.add_argument_group('More model combination options')
    group3 = parser.add_argument_group('Naive triangulation')

    group1.add_argument('action', metavar='ACTION', choices=["features_based","counts_based"],
                    help='What you want to do with the models. One of %(choices)s.')

    group1.add_argument('-s', metavar='DIRECTORY', dest='srcpvt',
                    help='model of the source and pivot, actually, it is going to be pivot-source')

    group1.add_argument('-t', metavar='DIRECTORY', dest='pvttgt',
                    help='model of pivot and target')

    group1.add_argument('-w', '--weight', dest='weight', type=str,
                    default='summation',
                    choices=["summation", "maximization"],
                    help='option to mixing identical pairs if the action is features_based, you have 2 choices: summation and maximization')

    group1.add_argument('-m', '--mode', type=str,
                    default="pst",
                    choices=["spt","pst"],
                    help='basic mixture-model algorithm. Default: %(default)s. Note: depending on mode and additional configuration, additional statistics are needed. Check docstring documentation of Triangulate_TMs() for more info.')

    group1.add_argument('-co', '--computation', dest='computation',
                    default="minimum",
                    choices=['minimum',"maximum","arithmetic-mean",'geometric-mean'],
                    help='choose to measures the co-occurrences if the action is counts_based, you have 4 options: minimum, maximum, arithmetic mean and geometric mean')

    group1.add_argument('-r', '--reference', type=str,
                    default=None,
                    help='File containing reference phrase pairs for cross-entropy calculation. Default interface expects \'path/model/extract.gz\' that is produced by training a model on the reference (i.e. development) corpus.')

    group1.add_argument('-o', '--output', type=str,
                    default="-",
                    help='Output file (phrase table). If not specified, model is written to standard output.')

    group1.add_argument('--output-lexical', type=str,
                    default=None,
                    help=('Not only create a combined phrase table, but also combined lexical tables. Writes to OUTPUT_LEXICAL.e2f and OUTPUT_LEXICAL.f2e, or OUTPUT_LEXICAL.counts.e2f in mode \'counts\'.'))

    group1.add_argument('-tmp', '--tmpdir', dest='tmp', type=str,
                    default=None,
                    help=('Temporary directory to put the intermediate phrase'))

    group2.add_argument('--write-phrase-penalty', action="store_true",
      help=("Include phrase penalty in phrase table"))

    group2.add_argument('--number_of_features', type=int,
                    default=4, metavar='N',
                    help=('Combine models with N + 1 features (last feature is constant phrase penalty). (default: %(default)s)'))


    group3.add_argument('--command', '--./tmtriangulate.py features_based -ps model1 -pt model2 -o output_phrasetable -t tempdir', action="store_true",
                    help=('If you wish to run the naive approach, the command above would work, in which: model1 = pivot-source model, model2 = pivot-target model'))


    return parser.parse_args()


# Configuration of Moses class
class Moses:
    ''' Moses interface for loading/writing models
        It keeps the value of src-pvt word count
    '''
    def __init__(self, number_of_features=4):
        self.number_of_features = number_of_features

        self.word_pairs_e2f = defaultdict(lambda: defaultdict(long))
        #self.word_pairs_f2e = defaultdict(lambda:defaultdict(long))

        self.word_count_e = defaultdict(long)
        self.word_count_f = defaultdict(long)

        self.phrase_count_f = None # name of the file with format tgt ||| src ||| count (sorted by tgt)
        self.phrase_count_e = None # name of the file with format src ||| tgt ||| count (sorted by tgt)

    def _compute_lexical_weight(self,src,tgt,alignments):
        '''
        compute the lexical weight in phrase table based on the co-occurrence of word count
        '''
        #TODO: This implementation is wrong, should I keep all the count in memory
        align_rev = defaultdict(lambda: [])
        alignment=defaultdict(lambda:[])

        phrase_src = src.split(b' ')
        phrase_tgt = tgt.split(b' ')

        # Value P(s|t) = pi(avg(w(si|ti)))
        weight_st = defaultdict(lambda: [])
        weight_ts = defaultdict(lambda: [])
        src_lst,tgt_lst = [],[]
        for src_id,tgt_id in alignments:
            weight_st[src_id].append(float(self.word_pairs_e2f[phrase_src[src_id]][phrase_tgt[tgt_id]])/self.word_count_f[phrase_tgt[tgt_id]])
            weight_ts[tgt_id].append(float(self.word_pairs_e2f[phrase_src[src_id]][phrase_tgt[tgt_id]])/self.word_count_e[phrase_src[src_id]])
            src_lst.append(src_id)
            tgt_lst.append(tgt_id)
        # Handle the unaligned words
        for idx in range(len(phrase_src)):
            if idx not in src_lst:
                weight_st[idx].append(float(self.word_pairs_e2f[phrase_src[idx]][b'NULL'])/self.word_count_f[b'NULL'])
        for idx in range(len(phrase_tgt)):
            if idx not in tgt_lst:
                weight_ts[idx].append(float(self.word_pairs_e2f[b'NULL'][phrase_tgt[idx]])/self.word_count_e[b'NULL'])

        # Compute the lexical
        lex_st = 1.0
        lex_ts = 1.0
        for src_id,val_lst in weight_st.iteritems():
            lex_st *= sum(val_lst)/len(val_lst)
        for tgt_id,val_lst in weight_ts.iteritems():
            lex_ts *= sum(val_lst)/len(val_lst)

        return lex_st, lex_ts

    #TODO: write the general lexical functions (both probability and count) instead of two functions
    #TODO: for the sake of parallelism, rewrite following functions to standards
    def _get_lexical(self,path,bridge,flag=0):
        ''' write the  lexical file
            named after: LexicalTranslationModel.pm->get_lexical
        '''
        sys.stderr.write("\nWrite the lexical files ")
        output_lex_prob_e2f = handle_file("{0}{1}.{2}".format(path,bridge,'e2f'), 'open', mode='w')
        output_lex_prob_f2e = handle_file("{0}{1}.{2}".format(path,bridge,'f2e'), 'open', mode='w')
        if flag == 1:
            output_lex_count_e2f = handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'e2f'), 'open', mode='w')
            output_lex_count_f2e = handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'f2e'), 'open', mode='w')

        count = 0
        for e,tgt_hash in self.word_pairs_e2f.iteritems():
            for f,val in tgt_hash.iteritems():
                if not count%100000:
                    sys.stderr.write(str(count)+'...')
                count+=1
                if flag == 1:
                    output_lex_count_e2f.write(b"%s %s %d %d\n" %(f,e,val,self.word_count_e[e]))
                    output_lex_count_f2e.write(b"%s %s %d %d\n" %(e,f,val,self.word_count_f[f]))
                output_lex_prob_e2f.write(b"%s %s %.7f\n" %(f,e,float(val)/self.word_count_e[e]))
                output_lex_prob_f2e.write(b"%s %s %.7f\n" %(e,f,float(val)/self.word_count_f[f]))

        handle_file("{0}{1}.{2}".format(path,bridge,'e2f'),'close',output_lex_prob_e2f,mode='w')
        handle_file("{0}{1}.{2}".format(path,bridge,'f2e'),'close',output_lex_prob_f2e,mode='w')
        if flag == 1:
            handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'e2f'),'close',output_lex_count_e2f,mode='w')
            handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'f2e'),'close',output_lex_count_f2e,mode='w')
        sys.stderr.write("Done\n")

    def _process_lexical_count_f(self,tempdir=None):
        ''' compute the count of target phrase, then write them down in format: src ||| tgt ||| count
            then sort the new file
        '''
        sys.stderr.write("\nProcess lexical count target: ")
        outsrc_file = "{0}/{1}.{2}".format(tempdir,"lexical_count","fe2f")
        outsrc = handle_file(outsrc_file, 'open', mode='w')
        if (not self.phrase_count_f): # do nothing for nothing
            sys.stderr.write("The phrase count is empty\n")
            return None
        count_tgt,key_tgt,reserve_lines = 0,None,[]
        count = 0
        for line in self.phrase_count_f:
            if not count%1000000:
                sys.stderr.write(str(count)+"...")
            count+=1
            line = line.strip().split(b' ||| ')
            if (key_tgt and key_tgt != line[0]):
                for l in reserve_lines:
                    outsrc.write(b"%s ||| %s ||| %i\n" %(l,key_tgt,count_tgt))
                reserve_lines,count_tgt = [],0
            count_tgt += int(line[2])
            key_tgt=line[0]
            reserve_lines.append(line[1])
        if (count_tgt):
            for l in reserve_lines:
                outsrc.write(b"%s ||| %s ||| %i\n" %(l,key_tgt,count_tgt))
        handle_file(outsrc_file, 'close', outsrc, mode='w')
        sys.stderr.write("Remove temporary target compact file {0}\n".format(self.phrase_count_f.name))
        os.remove(self.phrase_count_f.name)
        self.phrase_count_f = None
        # sort the lexical count by source
        src_sort_file = sort_file(outsrc_file,tempdir=tempdir)
        sys.stderr.write("Remove unsorted target compact file {0}\n".format(outsrc_file))
        os.remove(outsrc_file)

        return src_sort_file


    def _process_lexical_count_e(self,phrasefile,tempdir=None):
        ''' compute the count of source phrase, then write them down in the same format: src ||| tgt ||| count
            then sort the new file
        '''
        sys.stderr.write("Process lexical count source: ")
        outsrc_file = "{0}/{1}.{2}".format(tempdir,"lexical_count","ee2f")
        outsrc = handle_file(outsrc_file, 'open', mode='w')
        #insrc = handle_file(phrasefile,'open',mode='r')
        if (not phrasefile): # do nothing for nothing
            sys.stderr.write("The phrase count is empty\n")
            return None
        count_src,key_src,reserve_lines = 0,None,[]
        count = 0
        for line in phrasefile:
            if not count%1000000:
                sys.stderr.write(str(count)+"...")
            count+=1
            line = _load_line(line)
            if (key_src and key_src != line[0]):
                for l in reserve_lines:
                    outsrc.write(b"%s ||| %s ||| %i\n" %(key_src,l,count_src))
                reserve_lines,count_src = [],0
            count_src += int(line[4][2])
            key_src=line[0]
            reserve_lines.append(line[1])
        if (count_src):
            for l in reserve_lines:
                outsrc.write(b"%s ||| %s ||| %i\n" %(key_src,l,count_src))
        handle_file(outsrc_file, 'close', outsrc, mode='w')
        sys.stderr.write("No need for re-sorting the phrase\n")
        phrasefile.seek(0)
        # sort the lexical count by source
        # no need to be sort
        #src_sort_file = sort_file(outsrc_file,tempdir=tempdir)
        return handle_file(outsrc_file, 'open', mode='r')



# A set of global functions to support the usage of multi-threading

def _glob_get_lexical(word_pairs_e2f,word_count_e,word_count_f,path,bridge,flag=0):
    ''' write the  lexical file
            named after: LexicalTranslationModel.pm->get_lexical
    '''
    sys.stderr.write("\nWrite the lexical files ")
    output_lex_prob_e2f = handle_file("{0}{1}.{2}".format(path,bridge,'e2f'), 'open', mode='w')
    output_lex_prob_f2e = handle_file("{0}{1}.{2}".format(path,bridge,'f2e'), 'open', mode='w')
    if flag == 1:
        output_lex_count_e2f = handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'e2f'), 'open', mode='w')
        output_lex_count_f2e = handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'f2e'), 'open', mode='w')

    count = 0
    for e,tgt_hash in word_pairs_e2f.iteritems():
        for f,val in tgt_hash.iteritems():
            if not count%100000:
                sys.stderr.write(str(count)+'...')
            count+=1
            if flag == 1:
                output_lex_count_e2f.write(b"%s %s %d %d\n" %(f,e,val,word_count_e[e]))
                output_lex_count_f2e.write(b"%s %s %d %d\n" %(e,f,val,word_count_f[f]))
            output_lex_prob_e2f.write(b"%s %s %.7f\n" %(f,e,float(val)/word_count_e[e]))
            output_lex_prob_f2e.write(b"%s %s %.7f\n" %(e,f,float(val)/word_count_f[f]))

    handle_file("{0}{1}.{2}".format(path,bridge,'e2f'),'close',output_lex_prob_e2f,mode='w')
    handle_file("{0}{1}.{2}".format(path,bridge,'f2e'),'close',output_lex_prob_f2e,mode='w')
    if flag == 1:
        handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'e2f'),'close',output_lex_count_e2f,mode='w')
        handle_file("{0}{1}.{2}.{3}".format(path,bridge,"count",'f2e'),'close',output_lex_count_f2e,mode='w')
    sys.stderr.write("Done\n")
    return 1

def _glob_process_lexical_count_f(phrase_count_f,tempdir=None):
    ''' compute the count of target phrase, then write them down in format: src ||| tgt ||| count
            then sort the new file
    '''
    sys.stderr.write("\nProcess lexical count target: ")
    outsrc_file = "{0}/{1}.{2}".format(tempdir,"lexical_count","fe2f")
    outsrc = handle_file(outsrc_file, 'open', mode='w')
    if (not phrase_count_f): # do nothing for nothing
        sys.stderr.write("The phrase count is empty\n")
        return None
    count_tgt,key_tgt,reserve_lines = 0,None,[]
    count = 0
    for line in phrase_count_f:
        if not count%1000000:
            sys.stderr.write(str(count)+"...")
        count+=1
        line = line.strip().split(b' ||| ')
        if (key_tgt and key_tgt != line[0]):
            for l in reserve_lines:
                outsrc.write(b"%s ||| %s ||| %i\n" %(l,key_tgt,count_tgt))
            reserve_lines,count_tgt = [],0
        count_tgt += int(line[2])
        key_tgt=line[0]
        reserve_lines.append(line[1])
    if (count_tgt):
        for l in reserve_lines:
            outsrc.write(b"%s ||| %s ||| %i\n" %(l,key_tgt,count_tgt))
    handle_file(outsrc_file, 'close', outsrc, mode='w')
    sys.stderr.write("Remove temporary target compact file {0}\n".format(phrase_count_f.name))
    os.remove(phrase_count_f.name)
    phrase_count_f = None
    # sort the lexical count by source
    src_sort_file = sort_file_fix(outsrc_file,'phrase_count.f',tempdir=tempdir)
    sys.stderr.write("Remove unsorted target compact file {0}\n".format(outsrc_file))
    os.remove(outsrc_file)
    return 1

def _glob_process_lexical_count_e(phrasefile,tempdir=None):
    ''' compute the count of source phrase, then write them down in the same format: src ||| tgt ||| count
            then sort the new file
    '''
    sys.stderr.write("Process lexical count source: ")
    outsrc_file = "{0}/{1}.{2}".format(tempdir,"phrase_count","e")
    outsrc = handle_file(outsrc_file, 'open', mode='w')
    #insrc = handle_file(phrasefile,'open',mode='r')
    if (not phrasefile): # do nothing for nothing
        sys.stderr.write("The phrase count is empty\n")
        return None
    count_src,key_src,reserve_lines = 0,None,[]
    count = 0
    for line in phrasefile:
        if not count%1000000:
            sys.stderr.write(str(count)+"...")
        count+=1
        line = _load_line(line)
        if (key_src and key_src != line[0]):
            for l in reserve_lines:
                outsrc.write(b"%s ||| %s ||| %i\n" %(key_src,l,count_src))
            reserve_lines,count_src = [],0
        count_src += int(line[4][2])
        key_src=line[0]
        reserve_lines.append(line[1])
    if (count_src):
        for l in reserve_lines:
            outsrc.write(b"%s ||| %s ||| %i\n" %(key_src,l,count_src))
    handle_file(outsrc_file, 'close', outsrc, mode='w')
    sys.stderr.write("No need for re-sorting the phrase\n")
    phrasefile.seek(0)
    return 1

#merge the noisy phrase table
class Merge_TM():
    """This class take input as one noisy phrase table in which it consists of so many repeated lines.
       The output of this class should be one final clean phrase table

       The tasks which have to be done are:
        + Merge TM by summing them up
        + Merge TM by taking the maximum
        + Merge TM by co-occurrence
    """
    def __init__(self,model=None,
                      output_file=None,
                      mode='pst',
                      lang_src=None,
                      lang_target=None,
                      output_lexical=None,
                      action="features_based",
                      moses_interface=None,
                      weight='summation',
                      tempdir=None
                      ):

        self.mode = mode
        self.model = model # the model file
        self.output_file = output_file
        self.lang_src = lang_src
        self.lang_target = lang_target
        self.loaded = defaultdict(int)
        self.output_lexical = output_lexical
        self.action=action
        self.moses_interface=moses_interface
        self.weight=weight
        self.tempdir=tempdir
        # Parallelism, hack-ish way to run parallel
        # Damn python
        pool = Pool(processes=3)
        # get the path
        bridge = os.path.basename(self.output_file).replace("phrase-table","/lex").replace(".gz", "") # create the lexical associated with phrase table
        #self.moses_interface._get_lexical(os.path.dirname(os.path.realpath(self.output_file)), bridge,flag=0)
        lexc = Process(target=_glob_get_lexical, args=[self.moses_interface.word_pairs_e2f,self.moses_interface.word_count_e,self.moses_interface.word_count_f,os.path.dirname(os.path.realpath(self.output_file)), bridge,0])
        # handle the phrase count
        #self.phrase_count_f = self.moses_interface._process_lexical_count_f(tempdir=self.tempdir)
        lexf = Process(target=_glob_process_lexical_count_f, args=[self.moses_interface.phrase_count_f,self.tempdir])
        #self.phrase_count_e = self.moses_interface._process_lexical_count_e(self.model,tempdir=self.tempdir)
        lexe = Process(target=_glob_process_lexical_count_e, args=[self.model,self.tempdir])
        for p in [lexc,lexf,lexe]:
            p.start()
            sys.stderr.write(" --- process started at: {0} --- ".format(datetime.now()))
        for p in [lexc,lexf,lexe]:
            p.join()
            sys.stderr.write("--- process joined at: {0} --- ".format(datetime.now()))

        self.phrase_count_e=handle_file(os.path.normpath("{0}/{1}".format(self.tempdir,"phrase_count.e")),'open',mode='r')
        self.phrase_count_f=handle_file(os.path.normpath("{0}/{1}".format(self.tempdir,"phrase_count.f")),'open',mode='r')

    def _combine_TM(self,flag=False,prev_line=None):
        '''
        Get the unification of alignment
        merge multiple sentence into one
        '''
        prev_line = []
        sys.stderr.write("\nCombine Multiple lines by option: " + self.action + "\n")
        output_object = handle_file(self.output_file,'open',mode='w')
        sys.stderr.write("Start merging multiple lines ...")
        self._line_traversal = self._parallel_traversal

        # define the action
        if (self.action == 'features_based'):
            self._combine_lines = self._combine_sum
            self._recompute_features = self._recompute_features_Cohn
            if self.weight=='maximization':
                self._combine_lines = self._combine_max
                self._recompute_features = self._recompute_features_Cohn
        elif (self.action == 'counts_based'):
            self._combine_lines = self._combine_occ
            self._recompute_features = self._recompute_features_occ
        else:
            # by default, let say we take the cooccurrences and min
            self._combine_lines = self._combine_occ
            self._recompute_features = self._recompute_features_occ
        self._line_traversal(flag,prev_line,output_object)
        handle_file(self.output_file,'close',output_object,mode='w')

    def _parallel_traversal(self,flag=False,prev_line=None,output_object=None):
        ''' Traver through the phrase-table and the phrase compact file to compute the final file
        '''
        count = 0

        # keep the prev_line in memory until it break prev_line[0]
        for line,phrase_count_f,phrase_count_e in izip(self.model,self.phrase_count_f,self.phrase_count_e):
            if not count%1000000:
                sys.stderr.write(str(count)+'...')
            count+=1

            line = _load_line(line)
            phrase_count_ff = phrase_count_f.strip().split(b' ||| ')
            phrase_count_ee = phrase_count_e.strip().split(b' ||| ')

            if (line[0] != phrase_count_ff[0] or line[1] != phrase_count_ff[1]):
                sys.exit("Mismatch between phrase table and count table")
            else:
                line[4][0] = long(phrase_count_ff[2])
                line[4][1] = long(phrase_count_ee[2])

            if (prev_line):
                if (line[0] == prev_line[0] and line[1] == prev_line[1]):
                    # combine current sentence to previous sentence, return previous sentence
                    prev_line = self._combine_lines(prev_line, line)
                    continue
                else:
                    prev_line = self._recompute_features(prev_line)
                    # when you get out of the identical blog, start writing
                    outline = _write_phrasetable_file(prev_line)
                    output_object.write(outline)
                    prev_line = line
            else:
                # the first position
                prev_line = line
        if (len(prev_line)):
            prev_line = self._recompute_features(prev_line)
            outline = _write_phrasetable_file(prev_line)
            output_object.write(outline)
        sys.stderr.write("Done\n")

    def _recompute_features_Cohn(self,line):
        ''' Do nothing :)
        '''
        return line

    def _recompute_features_occ(self,line):
        '''
        Compute the value of a single according to the co-occurrence
        format: src ||| tgt ||| prob1 lex1 prob2 lex2 ||| align ||| c_t c_s c_s_t ||| |||
        '''
        coocc = float(line[4][2])
        count_s = line[4][1]
        count_t = line[4][0]

        # probability
        if (coocc == 0 and count_t == 0):
            line[2][0] = 0
        else:
            line[2][0] = coocc/count_t # p(s|t)
        if (coocc == 0 and count_s == 0):
            line[2][2] = 0
        else:
            line[2][2] = coocc/count_s # p(t|s)
        # lexical weight
        #TODO: pay attention to the change from lex1 to lex2
        line[2][1],line[2][3] = self.moses_interface._compute_lexical_weight(line[0],line[1],line[3])

        return line


    def _combine_occ(self,prev_line=None,cur_line=None):
        '''
        Calculate the value of combine occ by the co-occurrence
        rather than the probabilities
        '''
        # alignment
        alignment = []
        for pair in prev_line[3]+cur_line[3]:
            if (pair not in alignment):
                alignment.append(pair)
        prev_line[3] = alignment
        # count
        prev_line[4][2] += cur_line[4][2]
        return prev_line

    def _combine_sum(self,prev_line=None,cur_line=None):
        '''
        Summing up the probability
        Get the unification of alignment
        Get the sum of counts

        '''
        #TODO: reviews it
        for i in range(4):
            prev_line[2][i] += cur_line[2][i]
            prev_line[2][i] = min(prev_line[2][i], 1.0)
        # alignment
        alignment = []
        for pair in prev_line[3]+cur_line[3]:
            if (pair not in alignment):
                alignment.append(pair)
        prev_line[3] = alignment
        # count
        if (cur_line[4][0] != prev_line[4][0] or cur_line[4][1] != prev_line[4][1]):
            sys.exit("The numbers of current line and prev line are not the same")
        else:
            prev_line[4][2] += cur_line[4][2]
        return prev_line

    def _combine_max(self,prev_line=None,cur_line=None):
        '''
        Get the maximum the probability
        Get the unification of alignment
        Get the sum of counts
        '''
        # probability
        for i in range(4):
            prev_line[2][i] = max(prev_line[2][i], cur_line[2][i])
        # alignment
        alignment = []
        for pair in prev_line[3]+cur_line[3]:
            if (pair not in alignment):
                alignment.append(pair)
        prev_line[3] = alignment

        # count
        if (cur_line[4][0] != prev_line[4][0] or cur_line[4][1] != prev_line[4][1]):
            sys.exit("Incorrect numbers of counts")
        else:
            prev_line[4][2] += cur_line[4][2]
        return prev_line

class Triangulate_TMs():
    """This class handles the various options, checks them for sanity and has methods that define what models to load and what functions to call for the different tasks.
       Typically, you only need to interact with this class and its attributes.
    """

    def __init__(self,model1=None,
                      model2=None,
                      weight=None,
                      output_file=None,
                      mode='pst',
                      action=None,
                      computed=None,
                      tempdir=None,
                      number_of_features=4,
                      lang_src=None,
                      lang_target=None,
                      output_lexical=None,
                      write_phrase_penalty=None):

        self.mode = mode
        self.model1=model1
        self.model2=model2
        self.action = action # approaches
        self.output_file = output_file
        self.lang_src = lang_src
        self.lang_target = lang_target
        self.loaded = defaultdict(int)
        self.output_lexical = output_lexical
        self.tempdir=tempdir

        # It's possible to have the input in several modes: stp or tps
        self.inverted = None
        if mode not in ['pst','spt']:
            sys.stderr.write('Error: mode must be either "pst" or "spt"\n')
            sys.exit(1)
        elif mode == 'spt':
            self.inverted='src-pvt'

        self.number_of_features = int(number_of_features)

        # decide which option to estimate the new counts
        self.estimate_counts = get_minimum_counts
        if (self.action == 'counts_based'):
            if (computed == 'maximum'):
                self.estimate_counts = get_maximum_counts
            elif(computed == 'arithmetic-mean'):
                self.estimate_counts = get_arithmetic_mean
            elif(computed == 'geometric-mean'):
                self.estimate_counts = get_geometric_mean

        # The moses interface is to keep word counts
        self.moses_interface = Moses(4)

    def triangulate_standard(self,weight=None):
        ''' Triangulating two phrase tables and writing a temporary output file
        '''


        """write a new phrase table, based on existing weights of two other tables
           #NOTE: Indeed, all processes start here"""

        #1: formulate the input to pst mode
        model1 = (handle_file(os.path.join(self.model1,'model','phrase-table'), 'open', 'r'),1,1)
        model2 = (handle_file(os.path.join(self.model2,'model','phrase-table'), 'open', 'r'),1,2)
        model1, model2 = self._ensure_inverted(model1, model2)

        #2: prepare temporary files
        outtgt_file = os.path.normpath("{0}/{1}.{2}".format(self.tempdir,"lexical_count","f")) #lexical target count file
        outsrc_file = os.path.normpath("{0}/{1}.{2}".format(self.tempdir,"lexical_count","e")) # lexical source count file
        output_tgt = handle_file(outtgt_file, 'open', mode='w') # write one more file in format tgt ||| src ||| count_s+t
        output_src = handle_file(outsrc_file, 'open', mode='w') # write one more file in format src ||| tgt ||| count_s+t
        sys.stderr.write("Write compact lexical files: {0} and {1}\n" .format(outtgt_file, outsrc_file))
        output_object = handle_file(self.output_file,'open',mode='w')

        #3: Computing and writing new phrase tables
        #   The main function of triangulating phrase pairs
        self._get_features = self._get_features_Cohn
        sys.stderr.write('Incrementally loading and processing phrase tables...')
        # Start process phrase table
        self.phrase_match = defaultdict(lambda: []*3)
        self._phrasetable_traversal(model1=model1, model2=model2, prev_line1=None, prev_line2=None, deci=0, output_object=output_object,output_tgt=output_tgt,output_src=output_src)
        sys.stderr.write("Done\n")

        #4: Closing and cleaning temporary files
        handle_file(self.output_file,'close',output_object,mode='w')
        handle_file(outtgt_file,'close',output_tgt,mode='w')
        handle_file(outsrc_file,'close',output_src,mode='w')

        self.moses_interface.phrase_count_f = sort_file(outtgt_file,tempdir=self.tempdir)
        sys.stderr.write("Remove unsorted target compact file {0}\n" .format(outtgt_file))
        os.remove(outtgt_file)
        self.moses_interface.phrase_count_e = sort_file(outsrc_file,tempdir=self.tempdir)
        os.remove(outsrc_file)

    def _ensure_inverted(self, model1, model2):
        ''' This function handles the input mode
            All input styles have to be converted to \'pst\' style,
            which consists of pivot->source and pivot->target phrase tables
            It makes sure that the two phrase tables are sorted according
            to pivot phrases
        '''
        # If the mode is 'pst'
        if (not self.inverted):
            return (model1, model2)

        models=[]
        if (self.inverted == 'src-pvt'):
            models.append(model1)
        elif (self.inverted == 'tgt-pvt'):
            models.append(model2)
        elif (self.inverted == 'both'):
            models.append(model1)
            models.append(model2)
        else:
            # self.inverted = none or whatever
            return (model1, model2)

        for mod in models:
            outfile = NamedTemporaryFile(delete=False,dir=self.tempdir)
            output_contr = handle_file(outfile.name, 'open', mode='w')
            sys.stderr.write("Inverse model {0} > {1} ...".format(mod[0], outfile.name))
            count=0
            for line in mod[0]:
                if not count%100000:
                    sys.stderr.write(str(count)+'...')
                count+=1

                line = _load_line(line)
                # reversing src,tgt
                line[0],line[1] = line[1],line[0]
                # reverse probability
                line[2][0],line[2][2] = line[2][2],line[2][0]
                # reverse alignment
                for lid in range(len(line[3])):
                    line[3][lid][0],line[3][lid][1] =  line[3][lid][1],line[3][lid][0]
                # reverse count
                line[4][0],line[4][1] = line[4][0],line[4][1]

                outline = _write_phrasetable_file(line)
                output_contr.write(outline)
            handle_file(outfile.name,'close',output_contr,mode='w')
            tmpfile = sort_file(outfile.name,tempdir=self.tempdir)
            sys.stderr.write("Remove file: {0}\n" .format(outfile.name))
            os.remove(outfile.name)
            if (mod[2] == model1[2]):
                model1 = (tmpfile, model1[1], model1[2])
            elif (mod[2] == model2[2]):
                model2 = (tmpfile, model2[1], model2[2])
            sys.stderr.write("Done\n")
        return (model1, model2)

    def _phrasetable_traversal(self,model1,model2,prev_line1,prev_line2,deci,output_object,output_tgt,output_src):
        ''' A non-recursive way to read two models at the same time
            Notes: In moses phrase table, the longer phrase appears earlier than the short phrase
        '''
        line1 =  _load_line(model1[0].readline())
        line2 =  _load_line(model2[0].readline())
        count = 0
        while(1):
            if not count%1000000:
                sys.stderr.write(str(count)+'...')
            count+=1

            # Pivot phrases are identical
            if (self.phrase_match[0]):
                if (line1 and line1[0] == self.phrase_match[0]):
                    self.phrase_match[1].append(line1)
                    line1 =  _load_line(model1[0].readline())
                    continue
                elif (line2 and line2[0] == self.phrase_match[0]):
                    self.phrase_match[2].append(line2)
                    line2 = _load_line(model2[0].readline())
                    continue
                else:
                    self._combine_and_write(output_object,output_tgt,output_src)

            # End of file
            if (not line1 or not line2):
                self._combine_and_write(output_object,output_tgt,output_src)
                sys.stderr.write("Finish loading\n")
                return None

            # Compare the pivot phrases.
            # An exception if the long phrase contains the short phrase
            if (not self.phrase_match[0]):
                if (line1[0] == line2[0]):
                    self.phrase_match[0] = line1[0]
                elif (line1[0].startswith(line2[0])):
                    line1 = _load_line(model1[0].readline())
                elif (line2[0].startswith(line1[0])):
                    line2 = _load_line(model2[0].readline())
                elif (line1[0] < line2[0]):
                    line1 = _load_line(model1[0].readline())
                elif (line1[0] > line2[0]):
                    line2 = _load_line(model2[0].readline())

    def _combine_and_write(self,output_object,output_tgt,output_src):
        ''' Triangulating two phrases and write the new obtained phrases
        '''
        for phrase1 in self.phrase_match[1]:
            for phrase2 in self.phrase_match[2]:
                if (phrase1[0] != phrase2[0]):
                    sys.exit("Different pivot phrases\n")
                src, tgt = phrase1[1], phrase2[1]

                features = self._get_features(src, tgt, phrase1[2], phrase2[2])
                word_alignments = self._get_word_alignments(src, tgt, phrase1[3], phrase2[3])
                word_counts = self._get_cooccurrence_counts(src, tgt, phrase1[4], phrase2[4])
                outline = _write_phrasetable_file([src,tgt,features,word_alignments,word_counts])
                output_object.write(outline)
                output_tgt.write(b'%s ||| %s ||| %i\n' %(tgt,src,word_counts[2]))
                #output_src.write(b'%s ||| %s ||| %i\n' %(src,tgt,word_counts[2]))

                self._update_moses(src,tgt,word_alignments,word_counts)
        # reset the memory
        self.phrase_match = None
        self.phrase_match = defaultdict(lambda: []*3)


    def _update_moses(self, src, tgt, word_alignments, word_counts):
        ''' Update following variables: word counts e2f, f2e, phrase count e, f
        '''
        srcphrase = src.split(b' ')
        tgtphrase = tgt.split(b' ')
        tgt_lst = []
        src_lst = []
        for align in word_alignments:
            src_id,tgt_id=align
            self.moses_interface.word_pairs_e2f[srcphrase[src_id]][tgtphrase[tgt_id]] += word_counts[2]
            self.moses_interface.word_count_e[srcphrase[src_id]] += word_counts[2]
            #self.moses_interface.word_pairs_f2e[tgtphrase[tgt_id]][srcphrase[src_id]] += word_counts[2]
            self.moses_interface.word_count_f[tgtphrase[tgt_id]] += word_counts[2]
            tgt_lst.append(tgt_id)
            src_lst.append(src_id)

        # unaligned words
        for idx in range(len(tgtphrase)):
            if idx not in tgt_lst:
                self.moses_interface.word_pairs_e2f[b'NULL'][tgtphrase[idx]] += word_counts[2]
                self.moses_interface.word_count_e[b'NULL']+=word_counts[2]
                self.moses_interface.word_count_f[tgtphrase[idx]] += word_counts[2]
        # unaligned words
        for idx in range(len(srcphrase)):
            if idx not in src_lst:
                self.moses_interface.word_pairs_e2f[srcphrase[idx]][b'NULL'] += word_counts[2]
                self.moses_interface.word_count_f[b'NULL']+=word_counts[2]
                self.moses_interface.word_count_e[srcphrase[idx]] += word_counts[2]


    def _get_features_Cohn(self,src,target,feature1,feature2):
        ''' Compute the new features based on the posterior features
            Notes: in action 'features_based'
        '''
        phrase_features =  [0]*self.number_of_features
        phrase_features[0] = feature1[2] * feature2[0]
        phrase_features[1] = feature1[3] * feature2[1]
        phrase_features[2] = feature1[0] * feature2[2]
        phrase_features[3] = feature1[1] * feature2[3]

        return phrase_features

    def _get_word_alignments(self,src,target,phrase_ps,phrase_pt):
        """ Align source words and target words within the two phrases
            based on the pivot-source and pivot-target alignments
            The mismatched alignments are handled in _update_moses function
        """
        phrase_st = []
        for pvt_src in phrase_ps:
            for pvt_tgt in phrase_pt:
                if (pvt_src[0] == pvt_tgt[0]):
                    phrase_st.append((pvt_src[1],pvt_tgt[1]))
        return list(set(phrase_st))


    def _get_cooccurrence_counts(self,src,target,count1,count2):
        """ Estimate new co-occurrence counts based on the posterior co-occurrence counts
            Notes: in action 'counts_based'
        """
        word_count = [0]*3
        try:
            word_count[0] = count2[0]
            word_count[1] = count1[0]
            word_count[2] = self.estimate_counts(count1[2],count2[2])
        except:
            raise TypeError("Wrong format of co-occurrence counts")
        return word_count


def handle_file(filename,action,fileobj=None,mode='r'):
    """support reading/writing either from/to file, stdout or gzipped file"""

    if action == 'open':

        if mode == 'r':
            mode = 'rb'
        elif mode == 'w':
            mode = 'wb'

        if mode == 'rb' and not filename == '-' and not os.path.exists(filename):
            if os.path.exists(filename+'.gz'):
                filename = filename+'.gz'
            else:
                sys.stderr.write('Error: unable to open file. ' + filename + ' - aborting.\n')

                if 'counts' in filename and os.path.exists(os.path.dirname(filename)):
                    sys.stderr.write('For a weighted counts combination, we need statistics that Moses doesn\'t write to disk by default.\n')
                    sys.stderr.write('Repeat step 4 of Moses training for all models with the option -write-lexical-counts.\n')

                exit(1)

        if filename.endswith('.gz'):
            fileobj = gzip.open(filename,mode)

        elif filename == '-' and mode == 'wb':
            fileobj = sys.stdout

        else:
                fileobj = open(filename,mode)

        return fileobj

    elif action == 'close' and filename != '-':
        fileobj.close()


def sort_file(filename,tempdir=None):
    """Sort a file and return temporary file"""

    cmd = ['sort', filename]
    env = {}
    env['LC_ALL'] = 'C'
    if tempdir:
        cmd.extend(['-T',tempdir])

    outfile = NamedTemporaryFile(delete=False,dir=tempdir)
    sys.stderr.write('LC_ALL=C ' + ' '.join(cmd) + ' > ' + outfile.name + '\n')
    p = Popen(cmd,env=env,stdout=outfile.file)
    p.wait()

    outfile.seek(0)

    return outfile

def sort_file_fix(filename,newname,tempdir=None):
    """Sort a file and return temporary file with fix name"""

    cmd = ['sort', filename]
    env = {}
    env['LC_ALL'] = 'C'
    if tempdir:
        cmd.extend(['-T',tempdir])

    #outfile = NamedTemporaryFile(delete=False,dir=tempdir)
    outfile=open("{0}/{1}".format(tempdir,newname),mode='w')
    sys.stderr.write('LC_ALL=C ' + ' '.join(cmd) + ' > ' + outfile.name + '\n')
    p = Popen(cmd,env=env,stdout=outfile)
    p.wait()

    outfile.seek(0)

    return outfile



def dot_product(a,b):
    """calculate dot product from two lists"""

    # optimized for PyPy (much faster than enumerate/map)
    s = 0
    i = 0
    for x in a:
        s += x * b[i]
        i += 1

    return s

def get_minimum_counts(count1, count2):
    ''' get the mimimum occurrences between two occurrences
    '''
    return min(count1,count2)

def get_maximum_counts(count1, count2):
    ''' get the maximum occurrences between two occurrences
    '''
    return max(count1,count2)

def get_arithmetic_mean(count1, count2):
    ''' get arithmetic mean between two numbers
    '''
    return (count1+count2)/2
def get_geometric_mean(count1, count2):
    ''' get the geometric mean between two numbers
    '''
    return sqrt(count1*count2)

def _load_line(line):
    if (not line):
        return None
    ''' This function convert a string into an array of string and probability
        src ||| tgt ||| s|t s|t t|s t|s ||| align ||| countt counts countst ||| |||
    '''
    line = line.rstrip().split(b'|||')
    if line[-1].endswith(b' |||'):
        line[-1] = line[-1][:-4]
        line.append(b'')

    # remove the blank space
    line[0] = line[0].strip()
    line[1] = line[1].strip()

    # break the probability
    line[2]  = [float(i) for i in line[2].strip().split(b' ')]

    # break the alignment
    #TODO: Keep the alignment structure: [(1,1),(1,3),(2,3)]
    phrase_align = []
    for pair in line[3].strip().split(b' '):
        try:
            s,t = pair.split(b'-')
            s,t = int(s),int(t)
            phrase_align.append((s,t))
        except:
            pass
    line[3] = phrase_align
    line[4] = [long(float(i)) for i in line[4].strip().split(b' ')]
    if len(line[4]) < 2:
        sys.exit("the number of values in counting is not enough")

    return line

def _write_phrasetable_file(line):
    '''
    write the phrase table line
    '''
    # convert data to appropriate format
    # probability
    src,tgt,features,alignment,word_counts = line[:5]
    features = b' '.join([b'%.6g' %(f) for f in features])

    extra_space = b''
    if(len(alignment)):
        extra_space = b' '
    alignments = []
    for f in alignment:
        alignments.append(b"%i-%i" %(f[0],f[1]))
    alignments = b' '.join(alignments)

    word_counts = b' '.join([b'%.6g' %(f) for f in word_counts])

    outline = b"%s ||| %s ||| %s ||| %s%s||| %s ||| |||\n" %(src,tgt,features,alignments,extra_space,word_counts)
    return outline


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("no command specified. use option -h for usage instructions\n")

    elif sys.argv[1] == "test":
        test()

    else:
        args = parse_command_line()
        #1 Triangulate phrase pairs
        triangulator = Triangulate_TMs(weight=args.weight,
                               model1=args.srcpvt,
                               model2=args.pvttgt,
                               mode=args.mode,
                               output_file=os.path.normpath('/'.join([args.tmp, 'phrase-table'])),
                               action=args.action,
                               computed=args.computation,
                               output_lexical=args.output_lexical,
                               tempdir=args.tmp,
                               number_of_features=args.number_of_features,
                               write_phrase_penalty=args.write_phrase_penalty)

        triangulator.triangulate_standard()

        #2: Sort the temporary file
        tmpfile = sort_file(triangulator.output_file,tempdir=args.tmp)
        sys.stderr.write("Remove the unsorted phrase table {0}\n".format(triangulator.output_file))
        os.remove(triangulator.output_file)

        #3: Combine idential phrase pairs
        merger = Merge_TM(model=tmpfile,
                          output_file=args.output,
                          mode=triangulator.mode,
                          action=args.action,
                          moses_interface=triangulator.moses_interface,
                          weight=args.weight,
                          tempdir=args.tmp)
        merger._combine_TM()
