import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def load_all_data(data_path):
    
    X_train, y_train = [], []

    for sub in range (0,9):
        print(sub)
        path = data_path
        
        #load data from session 1 and 2 
        X1, y1 = load_data(path, sub+1, True)
        X2, y2 = load_data(path, sub+1, False)
        
        if (X_train == []):
            X_train = X1
            y_train = y1
            X_test = X2
            y_test = y2
        else:
            X_train = np.concatenate((X_train, X1), axis=0)
            y_train = np.concatenate((y_train, y1), axis=0)
            X_test = np.concatenate((X_test, X2), axis=0)
            y_test = np.concatenate((y_test, y2), axis=0)
 
    return X_train, y_train, X_test, y_test


def load_data(data_path, subject, training, all_trials = True):
	""" Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
	"""
    # Define MI-trials parameters
	n_channels = 3
	n_tests = 5*6*20 	
	window_Length = 7*250 

	class_return = np.zeros(n_tests)
	data_return = np.zeros((n_tests, n_channels, window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(data_path+'B0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(data_path+'B0'+str(subject)+'E.mat')
	a_data = a['data'] 
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_artifacts = a_data3[5]

		for trial in range(0,a_trial.size):
 			if(a_artifacts[trial] != 0 and not all_trials):
 			    continue
 			data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:3])
 			class_return[NO_valid_trial] = int(a_y[trial])
 			NO_valid_trial +=1


	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def get_data(path, isStandard = True):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)
    
    # Load and split the dataset into training and testing 
    X_train, y_train, X_test, y_test = load_all_data(path) 


    # Prepare training data 	
    N_tr, N_ch, _ = X_train.shape 
    X_train = X_train[:, :, t1:t2].reshape(N_tr, 1, N_ch, T)
    y_train_onehot = (y_train-1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    # Prepare testing data 
    N_test, N_ch, _ = X_test.shape 
    X_test = X_test[:, :, t1:t2].reshape(N_test, 1, N_ch, T)
    y_test_onehot = (y_test-1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)	
    
    # Standardize the data
    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train_onehot, X_test, y_test_onehot

