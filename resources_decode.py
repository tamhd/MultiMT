#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  This class implement a method to automatically extract the pair of language in one directory
#  it take the input as a "src" language and a "tgt" language and a "directory" of corpora
#  it outputs all the phrase pair necessary: direct parallel corpora, a list of all possible way to do triangulation and a list of monolingual corpora of the target language

from __future__ import division, unicode_literals
import sys
import os
import gzip
import argparse
import copy
import re
from math import log, exp
from collections import defaultdict
from operator import mul
from tempfile import NamedTemporaryFile
from tmcombine import Moses, Moses_Alignment, to_list

try:
    from itertools import izip
except:
    izip = zip

def parse_command_line():
    parser = argparse.ArgumentParser(description='Find the pair of data given in a directory, all the options will be implemented later')

    group1 = parser.add_argument_group('Main options')

    group1.add_argument('-s', metavar='DIRECTORY', dest='source',
                    help='source language that you want to translate from')
    group1.add_argument('-t', metavar='DIRECTORY', dest='target',
                    help='target language that you want to translate to')
    group1.add_argument('-c', metavar='DIRECTORY', dest='corporadir',
                    help='the directory which contains all corpora')
    group1.add_argument('-o', metavar='DIRECTORY', dest='output_file',
                    help='the output file')

    return parser.parse_args()


class Decode_Corpora():
    """This class handles the process of looking for a specific
       it also check the correctness of the bilingual corpus which they have the same line number or not
    """

    def __init__(self,source=None,
                      target=None,
                      corporadir=None,
                      output_file=None):

        ''' Initialize the value, list all the folder in the directory
        '''

        self.source = mode
        self.target = target
        self.cordir = corporadir
        self.output_file = output_file

    # A function I borrow from TM_Combine
    def _sanity_checks(self):
        """check if the corpora dir is clean, which mean:
           + the folder has to be in the alphabet order
           + in a bilingual directory, there must be an even number of file, in which each pair only different by the language in the folder
        """
        return True



    def _monolingual_find(self):
        """ find all the monolingual corpora for the language model
            return a list of files
            maybe another list of parallel corpora which could be used as monolingual corpora
        """
        return None

    def _bilingual_find(self):
        """ find all the bilingual corpora for the language model
            return a list of paired files
            (cs1,uk1), (cs2,uk2)
        """
        return None

    def _triangulation_find(self):
        """ find all the bilingual corpora for the language model
            return a dictionary
            dict -> {cs-en-uk}->[0] = [[cs1,en1],[cs2,en2]]
                              ->[1] = [[en,uk],[en1,uk1]]
            TODO: Think about the reverse pivot-* config
        """
        return None



# GLOBAL DEF
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



def dot_product(a,b):
    """calculate dot product from two lists"""

    # optimized for PyPy (much faster than enumerate/map)
    s = 0
    i = 0
    for x in a:
        s += x * b[i]
        i += 1

    return s



if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("no command specified. use option -h for usage instructions\n")

    elif sys.argv[1] == "test":
        test()

    else:
        args = parse_command_line()
        #initialize
        print "OK, Let's play!"
