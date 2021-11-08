# Learning-to-rank-ML-Model
Learning to rank is an algorithmic technique employing machine learning models to solve ranking problems. The modelling involves Gradient boosted methods like XGBoost and Lightgbm algorithm to solve the ranking problem. The dataset on which the gradient boosting algorithms is applied consists of parameters of a specific query like Query ID, Document ID, Page rank, Body terms etc. The code involved for solving ranking problem utilises Normalized Discounted Cumulative Gain (NDGC) metric to evaluate the query relevance. An NDGC of 0.6216 was achieved on the test set when the model is applied on the holdout.

# Files Involved
- Model.py : Python code containing modules involving data preparation, data modelling and generating run files
- Requirements.txt : Requirememts file containing python packages required for running the python code
- train.tsv : train data used for modelling
- test.tsv : test data used for testing the model.
