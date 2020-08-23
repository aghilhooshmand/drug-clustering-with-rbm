import tensorflow as tf
import sys
import csv
import numpy as np
%matplotlib inline
from matplotlib import pyplot as plt
import time
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
from sklearn import preprocessing
import keras
from sklearn.metrics import roc_auc_score
import collections
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras import backend as K
from keras import optimizers as optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import math
import gc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import plot_model
import pydot
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.utils import shuffle
from keras import metrics
import keras_metrics


Plr=[0.1,0.01,0.09,0.001,0.009]
Pepoch=[50,300,500,1000]
PBatchSize=[64,128]
PL2Regularization=[0,0.01,0.03,0.001,0.002]
     
#For each RBM in our list ------------------------------
learning_rate=Plr[1]
training_epochs=Pepoch[2]
batch_size=PBatchSize[1]

Weight=[]
HiddenBiase=[]
VisibleBiase=[]
# ModelLayer="E:/DATA/UniTese/Data/MACC_FP/result/98-9-8/154_154_154/NewRBM"

class BBRBM(object):
    
    def __init__(self, input_size, output_size):
        #Defining the hyperparameters
        self._input_size = input_size #Size of input
        self._output_size = output_size #Size of output
        #self.epochs = 5 #Amount of training iterations
        #self.learning_rate = 1.0 #The step used in gradient descent
        self.batchsize = 128 #The size of how much data will be used for training per sub iteration
        #Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float64) #Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float64) #Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float64) #Creates and initializes the visible biases with 0
        self.std=np.array([input_size],np.float64)
        self.var=np.array([input_size],np.float64)

    #Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        #Sigmoid 
       return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    #Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self,visible, hidden, w, vb):
         return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
    
    #Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs),dtype='float64')))

    #Training method for the model
    def train(self, X , learning_rate,epochs):
        #Create the placeholders for our parameters
        _w = tf.placeholder("float64",  [self._input_size, self._output_size])
        _hb = tf.placeholder("float64", [self._output_size])
        _vb = tf.placeholder("float64", [self._input_size])
        
        prv_w = np.random.random(size= [self._input_size, self._output_size]) #Creates and initializes the weights with 0
        prv_hb = np.random.random(size=[self._output_size]) # Creates and initializes the hidden biases with 0
        prv_vb = np.random.random(size=[self._input_size])  # Creates and initializes the visible biases with 0

        
        cur_w = np.random.random(size= [self._input_size, self._output_size])
        cur_hb = np.random.random(size=[self._output_size])
        cur_vb = np.random.random(size=[self._input_size])
        v0 = tf.placeholder("float64", [None, self._input_size])
        
        #Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(v0,h0, _w, _vb))
        h1 = self.sample_prob( self.prob_h_given_v(v1, _w, _hb))
        
        #Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        
        #Update learning rates for the layers
        update_w = _w + learning_rate *(positive_grad - negative_grad) / tf.to_double(tf.shape(v0)[0])
        update_vb = _vb +  learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb +  learning_rate * tf.reduce_mean(h0 - h1, 0)
        
        #Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))
        
        #Training loop
        errors=[]
        lr=learning_rate
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            #For each epoch
            for epoch in range(epochs):
                #For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize),range(self.batchsize,len(X), self.batchsize)):
                    batch = X[start:end]
                    #Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error=sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                errors.append( error)
                print ('Epoch: %d' % epoch,'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb
            Weight.append(self.w)
            HiddenBiase.append(self.hb)
            VisibleBiase.append(self.vb)
            #Print Chat of Train
            plt.plot(errors)
            plt.xlabel("Batch Number")
            plt.ylabel("Error")
#            plt.show()
            plt.savefig(str(ModelLayer)+"/NewRBM_macc_fp-rbm[" + str(self._input_size)+","+str(  self._output_size)+"]-Lr" +str(lr)+"-"+str(errors[-1])+"acc.png")
            plt.close()
            outF = open(str(ModelLayer)+"/NewRBM_PreTrain_Result.txt","a")
            ResultStr='\n'+'NEWRBM MACC'+'\t'+str(len(X))+'\t'+str(X.shape[1])+'\t'+str( self._input_size)+'\t'+str( self._output_size)+'\t'+str(lr)+'\t'+str(self.batchsize)+'\t'+str(epochs)+'\t'+str(errors[-1])
            outF.write(ResultStr)
            outF.close()
    
    #Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        probs=tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        out =tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs),dtype='float64')))
#        out=tf.where(out>0,1,0)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)         


########## Find Duplicate Column ######################
#######################################################
def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)
############ Drop Duplicate Column ##############################
def DropDuplicateColumn(dfObj):
    return dfObj.drop(columns=getDuplicateColumns(dfObj))
############ DF1 - DF2 ##############################
def Df1_Df2_Slow(df1,df2):
    df1=df1.reset_index(drop=True)
    df2=df2.reset_index(drop=True)
    ListIndexForDelete=[]
    for i in range(df1.shape[0]):
        count=0
        for j in range(df2.shape[0]):
            if (df1.iloc[i:i+1,].values==df2.iloc[j:j+1:].values).all():
                count=count+1
                break
        if count>0 :
            ListIndexForDelete.append(i)
    dfResult=df1.drop(ListIndexForDelete)  
    return dfResult 

def Df1_Df2_Fast(df1,df2):
    df = pd.merge(df1, df2, on=None, how='left', indicator='Exist')
    df['Exist'] = np.where(df.Exist == 'both', True, False)
    return df[df["Exist"]==False].iloc[:,:-1]   
#################################################################
##########  Thereshhold #########################################
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

import keras
from keras import backend as K
def threshold_binary_accuracy(y_true, y_pred):
    threshold = 0.5
    return K.mean(~K.equal(y_true, K.tf.cast(K.greater(threshold,y_pred), y_true.dtype)))

def PredictClass(y_pred,thresholdLow=0.5,thresholdHigh=1):
    return K.tf.cast((K.greater(y_pred,thresholdLow) & K.less(y_pred,thresholdHigh)),tf.int8)
        

All_Pre_Macc=pd.read_csv("E:/DATA/UniTese/Data/MultiMomdal_Zinc/Pretrain/All_Pre_Macc(PreTrain_Train_OtherData).csv")
 


Hormonizom=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Hormonizom_ColLower_0.1_Remove.csv")
Hormonizom=Hormonizom.iloc[:,4:]
Macc=pd.read_csv("file:///E:/DATA/UniTese/Data/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Macc_FP_ColLower_0.1_Remove.csv")
Macc=Macc.iloc[:,1:]
input_size_Macc = Macc.shape[1]
input_size_Harmonizome = Harmonizom.shape[1]

inpX=All_Pre_Macc

Weight=[]
HiddenBiase=[]

RbmList=[]
tf.reset_default_graph()


#All of RBM

print ('New bbRBM:')
BBRBM1=BBRBM(input_size_Macc, 78)
start_time=time.time()
BBRBM1.train(Macc,0.01,500)
TrainTime=time.time()-start_time
print('time is {0}'.format(TrainTime))
Out_First_Macc = BBRBM1.rbm_outpt(Macc)
rbm_list.append(BBRBM1)
del BBRBM1
gc.collect()

print ('New bbRBM:')
BBRBM2=BBRBM(input_size_Harmonizome, 1043)
start_time=time.time()
BBRBM2.train(Hormonizom,0.01,500)
TrainTime=time.time()-start_time
print('time is {0}'.format(TrainTime))
Out_First_Hormonizome = BBRBM2.rbm_outpt(input_size_Harmonizome)
rbm_list.append(BBRBM2)
del BBRBM2
gc.collect()

print ('New bbRBM:')
BBRBM3=BBRBM(Out_First_Hormonizome.shape[1], 104)
start_time=time.time()
BBRBM3.train(Out_First_Hormonizome,0.01,500)
TrainTime=time.time()-start_time
print('time is {0}'.format(TrainTime))
Out_First_Hormonizome = BBRBM3.rbm_outpt(Out_First_Hormonizome)
rbm_list.append(BBRBM3)
del BBRBM3
gc.collect()

#integrate
print ('New bbRBM:')
BBRBM4=BBRBM(Out_First_Hormonizome.shape[1]+Out_First_Macc.shape[1], 94)
start_time=time.time()
BBRBM4.train(pd.concat([Out_First_Macc,Out_First_Hormonizome],axis=1),0.01,500)
TrainTime=time.time()-start_time
print('time is {0}'.format(TrainTime))
Out_integrate = BBRBM4.rbm_outpt(pd.concat([Out_First_Macc,Out_First_Hormonizome],axis=1))
rbm_list.append(BBRBM4)
del BBRBM4
gc.collect()

#integrate
print ('New bbRBM:')
BBRBM5=BBRBM(Out_integrate.shape[1], 4)
start_time=time.time()
BBRBM5.train(Out_integrate,0.01,500)
TrainTime=time.time()-start_time
print('time is {0}'.format(TrainTime))
Out_Final = BBRBM5.rbm_outpt(Out_integrate)
rbm_list.append(BBRBM5)
del BBRBM5
gc.collect()

# ModelLayer="Macc(29074-104)_Hormonizom(29074-175)_L12_L14-L12-L12_clustering_By_Only_RBM"

p_lr=[0.1,0.01,0.09,0.001,0.009]
p_epoch=[1000]
p_batchsize=[64,128]
p_l2regularization=[0,0.01,0.02,0.03,0.001,0.002,0.003,0.004]
cdk=[1,3,5,7,10]
CdkDaynamic=[True,False]



Out=pd.DataFrame(Out_Final)
Out.insert(0,'Class',Hormonizom['Class'])
Out.insert(0,'CID',Hormonizom['CID'])
Out.insert(0,'Code',Hormonizom['Code'])
Out.to_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/"+str(ModelLayer)+"/All_Result.csv",index=False,sep="\t")


.del Out['PredictClass']
Out.insert(7,'PredictClass','-1')


def BinToDigit(d3,d2,d1,d0):
    return int((d3*pow(2,3))+(d2*pow(2,2))+(d1*pow(2,1))+(d0*pow(2,0)))

for i in range(Out.shape[0]):
    if Out.at[i,'Class']=='NoClass':
        Out.at[i,'Class']=-1
    

for i in range(Out.shape[0]):
    Out.at[i,'PredictClass']=BinToDigit(Out.at[i,'0'],Out.at[i,'1'],Out.at[i,'2'],Out.at[i,'3'])

Out = Out.astype({"Class":int})
Out = Out.astype({"PredictClass":int})

Out=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/"+str(ModelLayer)+"/All_Result.csv",delimiter="\t")

Out.groupby(['PredictClass']).count()
 
List=[]
for i in range(32):
    Cluster=Out[Out.PredictClass==i]
    for j in range(Out.shape[0]):
        


#-------- Save Model --------------------
import  pickle
f=open("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/"+str(ModelLayer)+"/Weight_Multimodal HC_MACC "+str(Macc.shape)+"_Hormonizom "+str(Hormonizom.shape)+"_epoch "+str(p_epoch[0])+"_lr "+str(p_lr[3])+"_batch"+str(p_batchsize[1])+"_CD "+str(cdk[0])+"_CdkDaynamic "+str(CdkDaynamic[0])+".file", "wb")
pickle.dump(Weight, f)
f=open("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/"+str(ModelLayer)+"/HiddenBias_Multimodal HC_MACC "+str(Macc.shape)+"_Hormonizom "+str(Hormonizom.shape)+"_epoch "+str(p_epoch[0])+"_lr "+str(p_lr[3])+"_batch"+str(p_batchsize[1])+"_CD "+str(cdk[0])+"_CdkDaynamic "+str(CdkDaynamic[0])+".file", "wb")
pickle.dump(HiddenBiase, f)

import  pickle
f=open("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/"+str(ModelLayer)+"/Weight_Multimodal HC_MACC "+str(Macc.shape)+"_Hormonizom "+str(Hormonizom.shape)+"_epoch "+str(p_epoch[0])+"_lr "+str(p_lr[3])+"_batch"+str(p_batchsize[1])+"_CD "+str(cdk[0])+"_CdkDaynamic "+str(CdkDaynamic[0])+".file", "wb")
pickle.dump(rbm_list, f)
f=open("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/"+str(ModelLayer)+"/HiddenBias_Multimodal HC_MACC "+str(Macc.shape)+"_Hormonizom "+str(Hormonizom.shape)+"_epoch "+str(p_epoch[0])+"_lr "+str(p_lr[3])+"_batch"+str(p_batchsize[1])+"_CD "+str(cdk[0])+"_CdkDaynamic "+str(CdkDaynamic[0])+".file", "wb")
pickle.dump(rbm_list, f)


#with open("E:/DATA/UniTese/Data/Morgan_Macc/Result/HiddenBiase_RBM_Zinc_"+str(All_Pre_Morgan.shape[0])+" (All_PreTrain_Train_NegativeForTrain)_"+"2204(Macc 154 + morgan 2048)-2204-2204-1000)"+"_MultimodalZinc-Morgan-Macc_"+str(p_epoch[4])+" epoch_"+str(p_lr[3])+" lr"+"_batch"+str(p_batchsize[1])+"_CD"+str(cdk[0])+"_CdkDaynamic "+str(CdkDaynamic[0])+".file", "wb") as f:
#    pickle.dump(HiddenBiase, f, pickle.HIGHEST_PROTOCOL)

#import pickle
#with open("E:/DATA/UniTese/Data/MultiMomdal_Zinc/Result/Weight_RBM_Zinc_"+str(All_Pre_Pubchem.shape[0])+" (All_PreTrain_Train_OtherData)_"+str(RbmListInf[0].num_vis)+"-"+str(RbmListInf[1].num_vis)+"-"+str(RbmListInf[2].num_vis)+"-"+str(RbmListInf[3].num_vis)+"_MultimodalZinc-Pubchem-Macc_"+str(p_epoch[4])+" epoch_"+str(p_lr[3])+" lr"+"_batch"+str(p_batchsize[1])+"_CD"+str(cdk[0])+"CdkDaynamic "+str(CdkDaynamic)+".file", "wb") as f:
#    pickle.dump(Weight, f, pickle.HIGHEST_PROTOCOL)
#with open("E:/DATA/UniTese/Data/MultiMomdal_Zinc/Result/HiddenBiase_RBM_Zinc_"+str(All_Pre_Pubchem.shape[0])+"(All_PreTrain_Train_OtherData)_"+str(RbmListInf[0].num_vis)+"-"+str(RbmListInf[1].num_vis)+"-"+str(RbmListInf[2].num_vis)+"-"+str(RbmListInf[3].num_vis)+"-"+"MultimodalZinc-Pubchem-Macc_"+str(p_epoch[4])+" epoch_"+str(p_lr[3])+" lr"+"_batch"+str(p_batchsize[1])+"_CD"+str(cdk[0])+"CdkDaynamic "+str(CdkDaynamic)+".file", "wb") as f:
#    pickle.dump(HiddenBiase, f, pickle.HIGHEST_PROTOCOL)




