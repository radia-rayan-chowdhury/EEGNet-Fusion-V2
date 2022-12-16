import time
from glob import glob
import numpy as np
import os
from run_type import RunType
from predict import predict_accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import EEGModels
from data_loader_BCI_2b import get_data

def train_test_model(model, model_name, X_train, y_train, X_test, y_test, multi_branch, nr_of_epochs, test_model=True):
    MODEL_LIST = glob('./model/*')
    new_model_name = './model/' + str(model_name) + '_' + str(len(MODEL_LIST)) + '.h5'
    print("New model name: " + new_model_name)
    acc = 0
    equals = []
    
    # Callbacks for saving best model, early stopping when validation accuracy does not increase and reducing
    # learning rate on plateau
    callbacks_list = [callbacks.ModelCheckpoint(new_model_name,
                                                save_best_only=True,
                                                monitor='val_loss'),
                      # callbacks.EarlyStopping(monitor='val_acc', patience=25),
                      callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]


    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    training_start = time.time()
    if multi_branch:
        history = model.fit([X_train, X_train, X_train, X_train, X_train], y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs,
                            validation_data=([X_test, X_test, X_test, X_test, X_test], y_test), verbose=False, callbacks=callbacks_list)
    else:
        history = model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_of_epochs,
                            validation_data=(X_test, y_test), verbose=False, callbacks=callbacks_list)
    training_total_time = time.time() - training_start
    print("Model {} total training time was {} seconds".format(model_name, training_total_time))
    print("That is {} seconds per sample".format(training_total_time/X_train.shape[0]))
    print("Train shape: {}. Test shape: {}".format(X_train.shape, X_test.shape))

    # test model predictions
    if test_model:
        model.load_weights(new_model_name)
        testing_start = time.time()
        acc, equals, preds = predict_accuracy(model, X_test, y_test, new_model_name, multi_branch=multi_branch)
        testing_total_time = time.time() - training_start
        print("Model {} total testing time was {} seconds".format(model_name, testing_total_time))
        print("That is {} seconds per sample".format(testing_total_time/X_test.shape[0]))
        
        rounded_labels = np.argmax(y_test, axis=1)

        precision_left = precision_score(rounded_labels, preds, average='binary', pos_label=0)
        print('Precision for left hand: %.3f' % precision_left)

        recall_left = recall_score(rounded_labels, preds, average='binary', pos_label=0)
        print('Recall for left hand: %.3f' % recall_left)

        f1_left = f1_score(rounded_labels, preds, pos_label=0, average='binary')
        print('F1-Score for left hand: %.3f' % f1_left)

        precision_right = precision_score(rounded_labels, preds, average='binary', pos_label=1)
        print('Precision for right hand: %.3f' % precision_right)

        recall_right = recall_score(rounded_labels, preds, average='binary', pos_label=1)
        print('Recall for right hand: %.3f' % recall_right)
        
        f1_right = f1_score(rounded_labels, preds, pos_label=1, average='binary')
        print('F1-Score for right hand: %.3f' % f1_right)

    return model, acc, equals


def getModel(model_name, use_cpu):
    if use_cpu:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')
        
    # Select the model
    if(model_name == 'EEGNet_fusion_V2'):
        model = EEGModels.EEGNet_fusion_V2(nb_classes = 2, Chans = 3, Samples=1125, cpu=use_cpu) 
    elif(model_name == 'EEGNet_fusion'):
        # Train using EEGNet Fusion: https://www.mdpi.com/2073-431X/9/3/72/htm
        model = EEGModels.EEGNet_fusion(nb_classes=2, Chans = 3, Samples = 1125, cpu=use_cpu)
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = EEGModels.EEGNet_classifier(nb_classes=2, Chans = 3, Samples = 1125, cpu=use_cpu) 
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = EEGModels.DeepConvNet(nb_classes=2, Chans = 3, Samples = 1125, cpu=use_cpu)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = EEGModels.ShallowConvNet(nb_classes=2, Chans = 3, Samples = 1125, cpu=use_cpu)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))
    return model

def run():
    # Get dataset path
    data_path = 'EEGMotorImagery-master/BCI Competition IV 2b/'
    
    # Create a folder to store the results of the experiment
    results_path = "EEGMotorImagery-master/results"
    
    # Create a new directory if it does not exist 
    if not os.path.exists(results_path):
      os.makedirs(results_path)   
      
    
    model_name = 'EEGNet_fusion_V2'
    
    #Get data 
    X_train, y_train, X_test, y_test = get_data(data_path, isStandard = True)
    
    # Create the model
    model = getModel(model_name, use_cpu = False)
    
    
     # splitting training/testing sets
    _model, acc, equals = train_test_model(model, model_name, X_train, y_train, X_test, y_test,
                                           multi_branch = True, nr_of_epochs = 100, test_model=True)
    _model.save('./model/' + str(model_name) + '_best.h5')
    

if __name__ == "__main__":
    run()
