Preamble
-------------------------------------------------------------------------------------
Required dependencies:

Python
***All code is with Python 2.7 (Python 3 is not supported)***
numpy >= 1.8
scipy >= 0.12
matplotlib >= 1.2

Any standard distribution of these packages should work

Matlab
command line functionality (ln -s PATH/TO/MATLAB matlab)
kmeans function
-------------------------------------------------------------------------------------
Install

all test scripts will run the compile the necessary auxiliary packages, which are included
-------------------------------------------------------------------------------------
Run

FOR SYNTHETIC DATA
to run an experiment, run the following command from the command line:

./run-batch

Currently, the parameters must be changed manually inside the test script (i.e. run-batch must be edited)
The parameters are as follows:
	generator: which model to generate data from
	
	The following are lists of parameters (one dataset will be generated for each
	combination):
		klist: number of topics
		nlist: number of documents
		llist: length of documents
		mlist: size of vocabulary
		alist: dirichlet parameter to control topics distributions for documents
		blist: dirichlet parameter to control word distributions for topics
To control the algorithms used for each experiment, you will need to modify the ./test script and set the "enableTest" field to true for each algorithm.

FOR REAL DATA
to run an experiment on real data, you will first need to convert it to a .json file. Several pre-processed datasets are included. An experiment can be run with the following command:

./test DATASET GENERATOR NUMTOPICS

if the dataset was generated according to the lda model, supply the argument "lda" to GENERATOR. Otherwise, supply "none."
-------------------------------------------------------------------------------------
Support

please direct any questions you may have to support@plusone.com
