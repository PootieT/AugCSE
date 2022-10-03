# Random Word Augmentation 
Randomly insert/ delete /split word in a sentence. A port-over augmentation from nlpaug
Author name: Zilu Tang 
Author email: zilutang@bu.edu
Author Affiliation: Boston University

## What type of a transformation is this?
This transformation acts like a perturbation to test robustness. Each word of the given text are randomly deleted/
inserted/split. 

## What tasks does it intend to benefit?
This perturbation would benefit all tasks which have a sentence/paragraph/document as input like text classification, 
text generation, etc. 

## Previous Work
Check original repo [nlpaug](https://github.com/makcedward/nlpaug)

## What are the limitations of this transformation?
Some augmentation can alter the meaning of the sentence, e.g. negations.
