# Amazon Reviews Clustering

The present project has been developed for the 2nd homework of the Big Data Computing course of the Master's Degree in Data Science.

It consists in processing a corpus of amazon reviews, vectorizing it, applying dimensionality reduction using Singular Value Decomposition and, finally, clustering the reviews into 2 clusters.

The accuracy reached is 0.85.  
For more detail check out the [report](BDC_HW2_report.pdf).

## HOW TO RUN THE JUPYTER NOTEBOOK

To run the jupyter notebook ```main.ipynb``` succesfully make sure to:
- have installed all the libraries/packages used, which are all specified in the first cell of the notebook. 
    - In particular, make sure to have the following ones: numpy, matplotlib, tqdm (for the progress bar), nltk, sklearn, scipy, wordcloud.

In the notebook there is also an implementation of TFIDF from scratch, which is fast to run but require A LOT of RAM.
So, I don't suggest you to run it if your notebook has less then 16 GB of RAM.

You should be able to run all the other cells of the notebook without problems.


## HOW TO RUN THE PYTHON FILE FROM COMMAND LINE

Included in the folder there is the file ```main.py``` which is a python script which can be run from command line using the following schema:  
```python .\main.py -c "data/corpus.txt" -l "data/labels.txt"```

The script also includes the HELP argument, which can be triggered by the following command:  
```python .\main.py -h ```.  
The HELP command will tell you what the different arguments are (```-c, -l, -sc```).  
Here is the description of the parameters:  
```-c, --corpus``` : Path of the .txt file containing the corpus.  
```-l, --labels``` : Path of the .txt file containing the labels.  
```--sc``` : Use a single cpu core.  

I suggest to DO NOT USE THE ```--sc``` ARGUMENT.  
The code is written to use by default the multiprocessing capabilities of your cpu to speed up execution.  
By using the ```--sc``` argument you are disabling multiprocessing, making the execution slower.

All the functions used in the ```main.py``` script can be found in the ```functions.py``` file.  
A more detailed description of what each snipped of code does can be found in the jupyter notebook. 
