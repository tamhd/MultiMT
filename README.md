
MultiMT
------

This is a project for translating one language to another language, with the support of a thid language(s).
It involves both the approaches to pivoting and the management techniques for available resources.

The first, and the most center to the project, is `tmtriangulate` - a tool for phrase table triangulation 

##### ABOUT `tmtriangulate`

This program handles the triangulation of Moses phrase tables, with 6 different options. 

##### REQUIREMENTS

The script requires Python >= 2.6. 

##### USAGE

`TmTriangulate` merges two phrase tables into one phrase table.

A command example: `./tmtriangulate.py features_based -s test/model1 -t test/model1`

This command will merge model1 with itself and estimate the feature values based on posterior probabilities, following the approach proposed by Cohn et al. 2007.

The basic command line: `./tmtriangulate.py [action] -s source-phrase-table -t target-phrase-table`

Until now, you can use two actions:

* `features_based`: Estimating new features from the posterior features. 

* `counts_based`: Re-estimating the co-occurrence counts

Each action is set to default with its best parameters. Typically, you have to specify a few parameters:

* mode (`-m`): accepts `pst` and `spt`, indicating the input as source-pivot phrase table or pivot-source phrase table

* computation (`-co`): specifies the scenario to triangulate the co-occurrence counts

* weight (`-w`): specifies the scenario to combine weights of identical phrase pairs

For further usage information, run `./tmcombine.py -h`

##### FURTHER NOTES

This project is under development! 

Author: Tam Hoang, Ondrej Bojar

If you have any comments, questions or suggestions, even jokes, feel free to send me an email at `tamhd1990 AT gmail DOT com`
