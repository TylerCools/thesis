
# SMemN2N Chatbot in Tensorflow

Implementation of [Source Awareness Memory End-to-End for Task-oriented Dialogue Learning]() with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](https://research.facebook.com/research/babi/) dataset. 

### Requirements

* tensorflow
* scikit-learn
* six
* scipy
* matplotlib


### Install and Run
```
pip install -r requirements.txt
python single_dialog.py
```
### Flags
```
This system has several flags to give as argument to the model. All the flags not speaking for itself are
discussed here: 
```
* OOV -> If this flag is used, the Out Of Vocabulary Knowledge Base is used
* source -> If this flag is used, Source Awareness will be used, otherwise the regular MemN2N
* wrong conversations -> All the conversations with wrong answers are outputted when using this flag
* error -> If this flag is true, the errors of the model are inspected. 
* acc each epoch -> Every epoch the test accuracy is calculated. This can be usefull for plots of accuracy
* acc ten epoch -> Every tenth epoch the test accuracy is calculated. This can be usefull for plots of accuracy
* conv wrong right -> A list of conversation numbers which were wrong and right predicted. This can be useful
					to check which conversations are predicted wrong in the MemN2N model and right in the SMemN2N model

### Extra information

* When the conv_wrong_right flag is used, the result can be used in the check.py function. This function compares two lists
to see which dialogues occur in both wrong MemN2N and right SMemN2N. The model needs to be ran twice to obtain both lists.
* The plots are automatically saved into the right folder. That is where you can find the images.

### Examples

Train the model
```
python single_dialog.py --train True --task_id 1 --source True
```

Testing the model
```
python single_dialog.py --train False --task_id 1 --source True 
```

Train and test the model at every epoch
```
python single_dialog.py --train True --task_id 1 --source True --acc_each_epoch True
```

Testing the model and see what the mistakes of the model are
```
python single_dialog.py --train False --task_id 1 --source True --error True --
```
### Results

Unless specified, the Adam optimizer was used.

The following params were used:
* epochs: 200
* learning_rate: 0.01
* epsilon: 1e-8
* embedding_size: 20


Task  |  Training Accuracy  |  Validation Accuracy  |  Test Accuracy	 
------|---------------------|-----------------------|--------------------
1     |  99.9	            |  99.1		            |  99.3				 
2     |  100                |  100		            |  99.9				 
3     |  96.1               |  71.0		            |  71.1				 
4     |  99.9               |  56.7		            |  57.2				 
5     |  99.9               |  98.4		            |  98.5				 
6     |  73.1               |  49.3		            |  40.6				 

### Authors

Tyler Cools -- University of Amsterdam -- Bachelor Artificial Intelligence