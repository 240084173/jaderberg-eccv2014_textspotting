# Deep Features for Text Spotting Code

## FOR MORE MODELS AND DATA SEE http://www.robots.ox.ac.uk/~vgg/research/text/

Models from the ECCV 2014 paper "Deep Features for Text Spotting" by Jaderberg et al.
http://www.robots.ox.ac.uk/~vgg/publications/2014/Jaderberg14/jaderberg14.pdf

You must cite this paper if you use this data or code.
```
#!bibtex

@InProceedings{Jaderberg14,
  author       = "Jaderberg, M. and Vedaldi, A. and Zisserman, A.",
  title        = "Deep Features for Text Spotting",
  booktitle    = "European Conference on Computer Vision",
  year         = "2014",
}
```


## Models
* Text/no-text classifer (models/detnet_layers.mat). 98.2% accuracy on ICDAR 2003.
* Case-insensitive character classifier (models/charnet_layers.mat). 91.0% accuracy on ICDAR 2003.
* Case-sensitive character classifier (models/casesnet_layers.mat). 86.8% accuracy on ICDAR 2003.
* ICDAR 2003 test-bigrams classifier (models/bigramic03net_layers.mat). 72.5% accuracy on ICDAR 2003.
* SVT test-bigrams classifier (models/bigramsvtnet_layers.mat).

## Data
* data/bigrams-train.mat - training data across all bigram classes.
* data/bigramsic03-train.mat - training data across ICDAR 2003 bigram classes only.
* data/case-insensitive-train.mat - case-insensitive character training data.
* data/case-sensitive-train.mat - case-sensitive character training data.
* data/icdar2003-bigrams-test.mat - test data across all bigram classes.
* data/icdar2003-bigramsic03-test.mat - test data across ICDAR 2003 bigram classes only.
* data/icdar2003-chars-test.mat - case-insensitive character test data.
* data/icdar2003-charscases-test.mat - case-sensitive character test data.

WARNING: The training datasets are comprised of data pulled from many sources, including the training datasets of other scene-text datasets (e.g. KAIST, ICDAR13, etc) so should only be used to train ICDAR03 and SVT.
    
## Setup
1. Edit matconvnet/Makefile to ensure MEX points to your matlab mex binary. Optinally ENABLE_GPU.
2. cd matconvnet/ && make

## Examples
1. fig_detmap.m
2. fig_charmap.m
3. reproduce_classifier_results.m

Max Jaderberg 2014
max@robots.ox.ac.uk
http://www.maxjaderberg.com


Thanks to Udit Roy for Makefile_linux