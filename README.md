
MultiMT
------

This is a project for translating one language to another language, with the support of a thid language(s).
It involves both the approaches to pivoting and the management techniques for available resources.

The first, and the most center to the project, is `tmtriangulate` - a tool for phrase table triangulation 

##### ABOUT `tmtriangulate`

This program handles the triangulation of Moses phrase tables, with 6 different options. 

##### REQUIREMENTS

The script requires Python >= 2.7.

The script has not yet been run on Windows.

##### USAGE

`TmTriangulate` merges two phrase tables into one phrase table.

A command example: `./tmtriangulate.py features_based -m pspt -s test/model1 -t test/model1`
This command will merge model1 with itself and estimate the feature values based on posterior probabilities.

The basic command line: `./tmtriangulate.py [action] -m [sppt] -s source-phrase-table -t target-phrase-table`

Until now, there are two actions, associated with two approaches to estimating values of the source-target phrase table:

* `features_based`: Computing the new probabilities from the component probabilities "Machine Translation by Triangulation: Making Effective Use of Multi-Parallel Corpora" (Cohn et al 2007)

* `counts_based`: Computing the new probabilities by approximating new co-occurrence counts "Improving Pivot-Based Statistical Machine Translation by Pivoting the Co-occurrence Count of Phrase Pairs" (Zhu et al 2014)

Each action is set to default with its best options. Typically, you have to specify a few parameters:

* mode (`-m`): indicates the direction of input phrase tables, i.e. source-pivot or pivot-source.

* computation (`-co`): specifies the scenario to triangulate the co-occurrence counts.

* weight (`-w`): specifies the scenario to combine weights of identical phrase pairs.

* source PT (`-s`): specifies the source phrase table or its directory with a given structure (dir/model/phrase-table)

* target PT (`-t`): specifies the target phrase table or its directory with a given structure (dir/model/phrase-table) 

For further usage information, run `./tmcombine.py -h`

##### FURTHER NOTES

This project is under development! 

Python multi-processing is automatically activated. There is no need for any configuration.

Author: Tam Hoang, Ondrej Bojar

If you have any comments, questions or suggestions, even jokes, feel free to send me an email at `tamhd1990 AT gmail DOT com`
