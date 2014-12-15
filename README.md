tmtriangulate - a tool for Moses translation with MultiMT

Author: Tam Hoang

ABOUT
-----

This program handles the combination of Moses phrase tables, but it is not tmcombine. 

From two phrase model src-pvt and pvt-tgt, the goal is a final src-tvt model

REQUIREMENTS
------------

The script requires Python >= 2.6.

In current source codes, I re-use a few classes written in TMCombine. Make sure that TMCombine could run here. 


OPTIONS
------

So far, we need to get the probabilities, lexical probabilities, alignment and word count. 

PROBABILITIES
 + 



After formalising the final table, we have to find a way to somehow prune it. Otherwise the triangulating phrase table is exceeding large.

 + Naive solution: we introduce a threshold to cut-off the phrase-table
 + 

FURTHER NOTES
-------------

 - A TM and reverse TM taken into consideration is: s.tm.cd277a36.20141114-1816 and s.tm.d2109fb2.20141214-1243
 - Next sample TMs are: s.tm.85e01946.20141214-1915 (czeng-europarl) and s.tm.98e7a68a.20141214-1915 (czeng-h1m), experiments shown that there is no different in building src-pvt or pvt-src table. jezzz

REFERENCES
----------

 - linear interpolation (naive): default
 - linear interpolation (modified): use options `--normalized` and `--recompute_lexweights`
 - weighted counts: use option `-m counts`
