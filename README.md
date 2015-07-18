
MultiMT
------

This is a project for translating one language to another language, with the support of a thid language(s).
This program handles the way to translate from one language to another via multiple languages in the middle. It involves either the approaches to triangulation and the management techniques towards available resources.

The first, and the most center to the project, is tmtriangulate - a tool for phrase-table triangulation 

##### ABOUT

This program handles the triangulation of Moses phrase tables, with 6 different options. 

##### REQUIREMENTS

The script requires Python >= 2.6. That's it, nothing more.

##### USAGE

TmTriangulate basically merges two phrase tables into one phrase table.

A command example: ./tmtriangulate.py features\_based -s test/model1 -t test/model1

This command triangulate model1 with itself, following the approach of Cohn

The basic command line: ./tmtriangulate.py [action] -s source-phrase-table -t target-phrase-table

Until now, you can use two actions:

* features\_based: Estimating new features from the posterior features. 

* counts\_based: Re-estimating the co-occurrence counts

Each action is set to default with its best parameters. Typically, you have to specify a few parameters:

* mode (-m): accepts \'pst\' and \'spt\', indicating the input as source-pivot phrase table or pivot-source phrase table

* computation (-co): specifies the scenario to triangulate the co-occurrence counts

* weight (-w): specifies the scenario to combine weights of identical phrase pairs

For further usage information, run ./tmcombine.py -h

##### FURTHER NOTES

This project is under development! 

Author: Tam Hoang, Ondrej Bojar

If you have any comments, questions or suggestions, even jokes, feel free to send me an email at tamhd1990@gmail.com
