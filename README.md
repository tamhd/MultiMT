
MultiMT
------

From a source to a target via multiple pivots

< Tam Hoang >

ABOUT
-----

This program handles the way to translate from one language to another via multiple languages in the middle. It involves either the approaches to triangulation and the management techniques towards available resources.

REQUIREMENTS
------------

The script requires Python >= 2.6.


USAGE
-----

The project is not a singled-system (yet) but a set of scripts: 'tmtriangulate.py' - building a final phrase table from two other phrase tables, 'resources\_decode.py' - finding the best way to ultilize the available resources.

For usage details, run ./tmtriangulate.py -h

For example: ./tmtriangulate.py combine\_given\_weights -ps model1 -pt model2 -o final-phrasetable -i none -t tempdir

There are three actions are provided : compute\_by\_occurrences, combine\_given\_weights and maximize\_given\_weights. 
The first scores a pair src-tgt by the new co-occurrences (default: minimum)
The second scores a pair src-tgt by all the possible ways to translate src to a pvt phrase then to tgt. 
The third scores a pair src-tgt by the best pvt phrase which bears the same meaning.

FURTHER NOTES
-------------

This project is under development! 
