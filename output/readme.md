# Data files
## vpinfo.nfo 
csv file that contains metainformation about participants, columns are: 
* 1 participant id
* 2 gender (1 - girl)
* 3 age in days
* 4 id of the person in charge of experiment (0=MS, 1=LW, 3=FK)
* 5 equal to 1 if caregiver gave consent with data publication
* 6 equal to 0 if the experiment took place on thursday or friday (there was no school on the day after tomorrow)
## res files in anonPublish
csv file provides information about participant's responses, columns are:
* 1 - output of code/exp6.py. Each integer provides the index of a response option. The order of element follows the presentation in exp6.py. The number of of questions was reduced due to covid19 pandemic for some subjects. The first entry provides the output-version index which determines the question-to-entry correspondence. See code/analyze.py for details.
* 2-6 - id of events that were respectively selected as possible, impossible, certain, likely and unlikely. To interpret the ID see table S1 in report
* 7 - id of events listed in order from most likely to least likely. 10 ranks were available to children. Events with equal rank are seperated by empty space. Some ranks may have been left without any events
* 8 - same as 7 after the experimenter had asked the child to check her answers
* 9 - position of events in centimeters along a line which went from least likely (0cm) to most likely (140cm)
* 10-15 - first element is children's explanation of the word while the second element is the exapmle of word's use provided by the child. The word were: possible, impossible, certain, likely, unlikely, more likely than
* 16 - experimenter comments and notes
