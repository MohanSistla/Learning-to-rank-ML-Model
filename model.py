import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lg
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# Function to split the "#" from header and loading dataset. Input of the function is the data file and output is formated data file

def strip_header(the_file):
  with open(the_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line
        else:
            break #stop when there are no more #
  the_header = header[1:].strip().split('\t')
  df = pd.read_csv(the_file,comment='#',names=the_header,sep='\t')
  return df

'''
 Function data_shuffling enables to group and shuffle the data before splitting the data into train and validation set in order to
 prevent the data from overfitting and under fitting.
 Input of the data is the train data set and output is the randomized train/validation indices to split data.
'''
def data_shuffling(data):
    gss = GroupShuffleSplit(test_size=.25, n_splits=5, random_state = 7).split(data, groups=data['QueryID'])
    X_train_inds, X_validation_inds = next(gss)
    return X_train_inds, X_validation_inds

'''
 Funtion data_train_prepare takes in the data file and corresponding randomized indexes and prepares 
 the data needed for training the ranking model. The function returns the original indexed data, train body, data label
 and the grouped data needed as per the ranking training model.
'''
def data_train_prepare(data,index):
    indexed_data= data.iloc[index]
    X_data = indexed_data.loc[:, ~indexed_data.columns.isin(['QueryID','Docid','Label'])]
    y_data = indexed_data.loc[:, indexed_data.columns.isin(['Label'])]
    group_data = indexed_data.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
    return indexed_data, X_data, y_data, group_data

'''
Funtion data_test_prepare takes in the data file and prepares the data for prediction. Input of the data is the test data
and output is the formatted data ready for data prediction.
'''
def data_test_prepare(data):
    test_df = strip_header('test.tsv')
    test_core_data = test_df.drop(['QueryID','Docid'],axis = 1)
    return test_core_data

'''
Funtion run_parameter_sweep performs a grid search on the model parameters in order to obtain the most suitable parameters to
train the data set and use it to predict the values associated with test set.
'''
def run_parameter_sweep(X_data,y_data,group):
    param_grid={
    'colsample_bytree':[0.3,0.9],
    'max_depth':[3,6],
    'subsample':[0.6, 1.0],
    'min_child_weight': [ 1,7],
    'n_estimators':[50,100],
    'eta': [0.05, 0.06]
    } 

    estimator = lg.LGBMRanker()
    cv = ShuffleSplit(n_splits=10, test_size=.25, random_state=0)

    model = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=8,scoring='neg_mean_squared_error')
    model.fit(X_data, y_data,group=group) 
    best_est = model.best_estimator_
    
    parameters = {
         'objective':"lambdarank",
         'metric':"ndcg",
         'random_state':42,
         'learning_rate':0.1,
         'colsample_bytree':best_est.colsample_bytree, 
         'eta':best_est.eta, 
         'max_depth':best_est.max_depth, 
         'n_estimators':best_est.n_estimators,  
         'subsample':best_est.subsample,
         'min_child_weight':best_est.min_child_weight
        }
    
    return parameters

'''
Funtion run_file_generation formats the predicted output into a proper data frame. Input of the function is the validation
or test data and repective prediction array. Output of the function is a properly format data frame returned as per
the requirements.
'''

def run_file_generation(data,prediction_array):
    vali_df = data.copy()
    vali_df['Score'] = prediction_array.tolist()
    df_vali_filtered = vali_df[['QueryID','Docid','Score']]
    return df_vali_filtered

'''
Main Function
'''

if __name__ == "__main__":
    
    train_df = strip_header('train.tsv')
    test_df =  strip_header('test.tsv')
    shuffleData = data_shuffling(train_df)
    train_data = data_train_prepare(train_df,shuffleData[0])
    validation_data = data_train_prepare(train_df,shuffleData[1])
    
    # Setting sweep equal to False takes into the account the parameters responsible to achieve good ndgc score on test data
    sweep = False
    
    if sweep == False:  
        
        parameters = {
            'objective':"lambdarank",
            'metric':"ndcg",
            'random_state':42,
            'learning_rate':0.09,
            'colsample_bytree':0.3,  
            'max_depth':4, 
            'n_estimators':50,  
            'subsample':0.4,
            'min_child_weight':1,
            'num_leaves':31
        }

    else:   # Takes into the parameters returned by grid search which takes few hours to run
        parameters = run_parameter_sweep(train_data[1],train_data[2],train_data[3])
        
    model = lg.LGBMRanker(**parameters)
    model.fit(X=train_data[1], y=train_data[2], group=train_data[3])
    pred = model.predict(validation_data[1])
    validation_run_file = run_file_generation(validation_data[0],pred)
#   validation_run_file.to_csv("trainScore.tsv",sep = "\t",index = False,header = False) # To create Validation run
    test = test_df.drop(['QueryID','Docid'],axis = 1)
    pred_test_file = model.predict(test)
    test_run_file = run_file_generation(test_df,pred_test_file)
    test_run_file.to_csv("A2.tsv",sep = "\t",index = False,header = False)
