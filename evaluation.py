from models import *
import os
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from idhp_data import *
import SimpleITK as sitk
import cv2
import numpy as np
import math
import copy

def policy_val(t, yf, q_t0, q_t1, q_t2, compute_policy_curve=False):
    # if np.any(np.isnan(eff_pred)):
        # return np.nan, np.nan
    q_cat = np.concatenate((q_t0, q_t1),1)
    q_cat = np.concatenate((q_cat, q_t2),1)
    policy = np.argmax(q_cat,1)
    policy = policy[:,np.newaxis]
    #policy = np.ones(policy.shape)
    t0_overlap = (policy==t)*(t==0)
    t1_overlap = (policy==t)*(t==1)
    t2_overlap = (policy==t)*(t==2)
    
    if np.sum(t0_overlap) == 0:
        t0_value = 0
    else: 
        t0_value = np.mean(yf[t0_overlap])
        
    if np.sum(t1_overlap) == 0:
        t1_value = 0
    else: 
        t1_value = np.mean(yf[t1_overlap])
        
    if np.sum(t2_overlap) == 0:
        t2_value = 0
    else: 
        t2_value = np.mean(yf[t2_overlap])
    
    pit_0 = np.sum(policy==0)/len(t)
    pit_1 = np.sum(policy==1)/len(t)
    pit_2 = np.sum(policy==2)/len(t)
    policy_value = pit_0*t0_value + pit_1*t1_value + pit_2*t2_value

    
    return policy_value

def factual_acc(t, yf, q_t0, q_t1, q_t2):

    q_t0_ = copy.copy(q_t0)
    q_t1_ = copy.copy(q_t1)
    q_t2_ = copy.copy(q_t2)


    q_t0_[q_t0_>=0.5] = 1
    q_t0_[q_t0_<0.5] = 0
    
    q_t1_[q_t1_>=0.5] = 1
    q_t1_[q_t1_<0.5] = 0
    
    q_t2_[q_t2_>=0.5] = 1
    q_t2_[q_t2_<0.5] = 0
    
    accuracy_0 = np.sum(q_t0_[t==0]==yf[t==0])/len(yf[t==0])
    accuracy_1 = np.sum(q_t1_[t==1]==yf[t==1])/len(yf[t==1])
    accuracy_2 = np.sum(q_t2_[t==2]==yf[t==2])/len(yf[t==2])
    
    #print("Factual accuracy of t0:", accuracy_0)
    #print("Factual accuracy of t1:", accuracy_1)
    #print("Factual accuracy of t2:", accuracy_2)
    
    return accuracy_0,accuracy_1,accuracy_2

def factual_acc_(t, yf, q_t0, q_t1, q_t2):



    q_t0[q_t0>=0.5] = 1
    q_t0[q_t0<0.5] = 0
    
    q_t1[q_t1>=0.5] = 1
    q_t1[q_t1<0.5] = 0
    
    q_t2[q_t2>=0.5] = 1
    q_t2[q_t2<0.5] = 0
    
    accuracy_0 = np.sum(q_t0[t==0]==yf[t==0])/len(yf[t==0])
    accuracy_1 = np.sum(q_t1[t==1]==yf[t==1])/len(yf[t==1])
    accuracy_2 = np.sum(q_t2[t==2]==yf[t==2])/len(yf[t==2])
    
    #print("Factual accuracy of t0:", accuracy_0)
    #print("Factual accuracy of t1:", accuracy_1)
    #print("Factual accuracy of t2:", accuracy_2)
    
    return accuracy_0,accuracy_1,accuracy_2

def factual_auc(t, yf, q_t0, q_t1, q_t2):

    auc_0 = metrics.roc_auc_score(yf[t==0],q_t0[t==0])
    auc_1 = metrics.roc_auc_score(yf[t==1],q_t1[t==1])
    auc_2 = metrics.roc_auc_score(yf[t==2],q_t2[t==2])
    
    return auc_0,auc_1,auc_2
    
def policy_risk_multi(t, yf, q_t0, q_t1, q_t2):
    policy_value = policy_val(t, yf, q_t0, q_t1, q_t2)
    policy_risk = 1 - policy_value
    return policy_risk
  
def ate_error_0_1(t, yf, eff_pred):
    att = np.mean(yf[t==0]) - np.mean(yf[t==1])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)
    
def ate_error_0_2(t, yf, eff_pred):
    att = np.mean(yf[t==0]) - np.mean(yf[t==2])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)
    
def ate_error_1_2(t, yf, eff_pred):
    att = np.mean(yf[t==1]) - np.mean(yf[t==2])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)
    
def _split_output(yt_hat, t, y, y_scaler, x, is_train=False):
    """
        Split output into dictionary for easier use in estimation
        Args:
            yt_hat: Generated prediction
            t: Binary treatment assignments
            y: Treatment outcomes
            y_scaler: Scaled treatment outcomes
            x: Covariates
            index: Index in data

        Returns:
            Dictionary of all needed data
    """
    traumatic = x[:,3]
    traumatic_index = np.where(traumatic==1)
    yt_hat = yt_hat[traumatic_index]
    t = t[traumatic_index]
    y = y[traumatic_index]
    y_scaler = y_scaler[traumatic_index]
    x = x[traumatic_index]
    
    yt_hat = yt_hat
    q_t0 = yt_hat[:, 0].reshape(-1, 1).copy()
    q_t1 = yt_hat[:, 1].reshape(-1, 1).copy()
    q_t2 = yt_hat[:, 2].reshape(-1, 1).copy()
    g = yt_hat[:, 3:6].copy()

    treatment_predicted = np.argmax(g,1)

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y.copy()
    var = "average propensity for t0: {} and t1: {} and t2: {}".format(g[:,0][t.squeeze() == 0.].mean(),
                                                                        g[:,1][t.squeeze() == 1.].mean(),g[:,2][t.squeeze() == 2.].mean())
    
    #auc_0,auc_1,auc_2 = factual_auc(t, y, q_t0, q_t1, q_t2)
    auc_0,auc_1,auc_2 = 0,0,0
    accuracy_0,accuracy_1,accuracy_2 = factual_acc(t, y, q_t0, q_t1, q_t2)
    
    
    q_cat = np.concatenate((q_t0, q_t1),1)
    q_cat = np.concatenate((q_cat, q_t2),1)
    
    #policy = np.argmax(q_cat,1)
    
    
    
    return {'ave propensity for t0': g[:,0][t.squeeze() == 0.].mean(), 'ave propensity for t1': g[:,1][t.squeeze() == 1.].mean(),
    'ave propensity for t2': g[:,2][t.squeeze() == 2.].mean(), 'Policy Risk': policy_risk_multi(t, y, q_t0, q_t1, q_t2), 
    'Ate_Error_0_1': ate_error_0_1(t, y, q_t0 - q_t1), 'Ate_Error_0_2': ate_error_0_2(t, y, q_t0 - q_t2),
    'Ate_Error_1_2': ate_error_1_2(t, y, q_t1 - q_t2), 'Treatment accuracy': np.sum(treatment_predicted==t.squeeze())/treatment_predicted.shape[0], 
    'Treatment policy': np.argmax(q_cat,1), 'Treatment prediction': treatment_predicted, 'Treatment label': t.squeeze().astype(int),'accuracy_0':accuracy_0,
    'accuracy_1':accuracy_1,'accuracy_2':accuracy_2,'auc_0':auc_0,
    'auc_1':auc_1,'auc_2':auc_2}


average_propensity_for_t0 = []
average_propensity_for_t1 = []
average_propensity_for_t2 = []
policy_risk = []
test_ate_error_0_1 = []
test_ate_error_0_2 = []
test_ate_error_1_2 = []
treatment_accuracy = []
treatment_policy=np.array([])
treatment_prediction=np.array([])
treatment_label=np.array([])
test_factual_accuracy_of_t0 = []
test_factual_accuracy_of_t1 = []
test_factual_accuracy_of_t2 = []


train_average_propensity_for_t0 = []
train_average_propensity_for_t1 = []
train_average_propensity_for_t2 = []
train_policy_risk = []
train_ate_error_0_1 = []
train_ate_error_0_2 = []
train_ate_error_1_2 = []
train_treatment_accuracy = []
train_factual_accuracy_of_t0 = []
train_factual_accuracy_of_t1 = []
train_factual_accuracy_of_t2 = []
test_factual_auc_of_t0 = []
test_factual_auc_of_t1 = []
test_factual_auc_of_t2 = []
key_word = 'Treatment accuracy'
key_word4 = 'Policy Risk'
key_word5 = 'accuracy_0'
key_word6 = 'accuracy_1'
key_word7 = 'accuracy_2'

key_word1 = 'Ate_Error_0_1'
key_word2 = 'Ate_Error_0_2'
key_word3 = 'Ate_Error_1_2'

for validation_index in range(10):
    best_evaluation = 1.
    train_outputs_best = {}
    test_outputs_best = {}
    for epoch in range(600,1500,10):
        test_results = np.load("./results_save/cli/{}_fold_{}_epoch_test.npz".format(validation_index, epoch), allow_pickle=True)
        train_results = np.load("./results_save/cli/{}_fold_{}_epoch_train.npz".format(validation_index, epoch), allow_pickle=True)
        
        yt_hat_test, t_test, y_test, y, x_test = test_results['yt_hat_test'], test_results['t_test'], test_results['y_test'], \
        test_results['y'], test_results['x_test']
        yt_hat_train, t_train, y_train, y, x_train = train_results['yt_hat_train'], train_results['t_train'], train_results['y_train'], \
        train_results['y'], train_results['x_train']
        
        test_outputs = _split_output(yt_hat_test, t_test, y_test, y, x_test, is_train=False)
        train_outputs = _split_output(yt_hat_train, t_train, y_train, y, x_train, is_train=True)
        #test_outputs = test_outputs['arr_0'].item()
        #train_outputs = train_outputs['arr_0'].item()
        #if test_outputs[key_word] <= best_evaluation and epoch>=500:
        if (test_outputs[key_word1]+test_outputs[key_word2]+test_outputs[key_word3]+test_outputs[key_word4]+(1-test_outputs[key_word5])+(1-test_outputs[key_word6])+(1-test_outputs[key_word7]))/7 <= best_evaluation and epoch>=500:
            test_outputs_best = test_outputs
            #best_evaluation = test_outputs[key_word]
            best_evaluation = (test_outputs[key_word1]+test_outputs[key_word2]+test_outputs[key_word3]+test_outputs[key_word4]+(1-test_outputs[key_word5])+(1-test_outputs[key_word6])+(1-test_outputs[key_word7]))/7
        
        train_outputs_best = train_outputs
        # if (train_outputs[key_word1]+train_outputs[key_word2]+train_outputs[key_word3]+train_outputs[key_word4]+(1-train_outputs[key_word5])+(1-train_outputs[key_word6])+(1-train_outputs[key_word7]))/7 <= best_evaluation and epoch>=500:
            # train_outputs_best = train_outputs
            
            # #best_evaluation = test_outputs[key_word]
            # best_evaluation = (train_outputs[key_word1]+train_outputs[key_word2]+train_outputs[key_word3]+train_outputs[key_word4]+(1-train_outputs[key_word5])+(1-train_outputs[key_word6])+(1-train_outputs[key_word7]))/7

        
    print("==========Best test results for the {} fold==========".format(validation_index))
    print("average propensity for t0: {} and t1: {} and t2: {}".format(test_outputs_best['ave propensity for t0'],test_outputs_best['ave propensity for t1'],
    test_outputs_best['ave propensity for t2']))
    print("Policy Risk:", test_outputs_best['Policy Risk'])
    print("Ate_Error_0_1:", test_outputs_best['Ate_Error_0_1'])
    print("Ate_Error_0_2:", test_outputs_best['Ate_Error_0_2'])
    print("Ate_Error_1_2:", test_outputs_best['Ate_Error_1_2'])
    print("Treatment accuracy:", test_outputs_best['Treatment accuracy'])
    print("Treatment policy    :",test_outputs_best['Treatment policy'])
    print("Treatment prediction:",test_outputs_best['Treatment prediction'])
    print("Treatment label     :",test_outputs_best['Treatment label'])
    print("Factual accuracy of t0:", test_outputs_best['accuracy_0'])
    print("Factual accuracy of t1:", test_outputs_best['accuracy_1'])
    print("Factual accuracy of t2:", test_outputs_best['accuracy_2'])
    print("Factual auc of t0:", test_outputs_best['auc_0'])
    print("Factual auc of t1:", test_outputs_best['auc_1'])
    print("Factual auc of t2:", test_outputs_best['auc_2'])
    print("==========Best train results for the {} fold==========".format(validation_index))
    print("average propensity for t0: {} and t1: {} and t2: {}".format(train_outputs_best['ave propensity for t0'],train_outputs_best['ave propensity for t1'],
    train_outputs_best['ave propensity for t2']))
    print("Policy Risk:", train_outputs_best['Policy Risk'])
    print("Ate_Error_0_1:", train_outputs_best['Ate_Error_0_1'])
    print("Ate_Error_0_2:", train_outputs_best['Ate_Error_0_2'])
    print("Ate_Error_1_2:", train_outputs_best['Ate_Error_1_2'])
    print("Treatment accuracy:", train_outputs_best['Treatment accuracy'])
    print("Factual accuracy of t0:", train_outputs_best['accuracy_0'])
    print("Factual accuracy of t1:", train_outputs_best['accuracy_1'])
    print("Factual accuracy of t2:", train_outputs_best['accuracy_2'])
    print("Factual auc of t0:", train_outputs_best['auc_0'])
    print("Factual auc of t1:", train_outputs_best['auc_1'])
    print("Factual auc of t2:", train_outputs_best['auc_2'])
    print("====================================================")
    average_propensity_for_t0.append(test_outputs_best['ave propensity for t0'])
    average_propensity_for_t1.append(test_outputs_best['ave propensity for t1'])
    average_propensity_for_t2.append(test_outputs_best['ave propensity for t2'])
    policy_risk.append(test_outputs_best['Policy Risk'])
    test_ate_error_0_1.append(test_outputs_best['Ate_Error_0_1'])
    test_ate_error_0_2.append(test_outputs_best['Ate_Error_0_2'])
    test_ate_error_1_2.append(test_outputs_best['Ate_Error_1_2'])
    treatment_accuracy.append(test_outputs_best['Treatment accuracy'])
    test_factual_accuracy_of_t0.append(test_outputs_best['accuracy_0'])
    test_factual_accuracy_of_t1.append(test_outputs_best['accuracy_1'])
    test_factual_accuracy_of_t2.append(test_outputs_best['accuracy_2'])
    test_factual_auc_of_t0.append(test_outputs_best['auc_0'])
    test_factual_auc_of_t1.append(test_outputs_best['auc_1'])
    test_factual_auc_of_t2.append(test_outputs_best['auc_2'])
    treatment_policy=np.concatenate((treatment_policy,test_outputs_best['Treatment policy']),0)
    treatment_prediction=np.concatenate((treatment_prediction,test_outputs_best['Treatment prediction']),0)
    treatment_label=np.concatenate((treatment_label,test_outputs_best['Treatment label']),0)

    train_average_propensity_for_t0.append(train_outputs_best['ave propensity for t0'])
    train_average_propensity_for_t1.append(train_outputs_best['ave propensity for t1'])
    train_average_propensity_for_t2.append(train_outputs_best['ave propensity for t2'])
    train_policy_risk.append(train_outputs_best['Policy Risk'])
    train_ate_error_0_1.append(train_outputs_best['Ate_Error_0_1'])
    train_ate_error_0_2.append(train_outputs_best['Ate_Error_0_2'])
    train_ate_error_1_2.append(train_outputs_best['Ate_Error_1_2'])
    train_factual_accuracy_of_t0.append(train_outputs_best['accuracy_0'])
    train_factual_accuracy_of_t1.append(train_outputs_best['accuracy_1'])
    train_factual_accuracy_of_t2.append(train_outputs_best['accuracy_2'])
    train_treatment_accuracy.append(train_outputs_best['Treatment accuracy'])

print("==========Average best test results==========")
print("average propensity for t0: {} and t1: {} and t2: {}".format(np.mean(average_propensity_for_t0),np.mean(average_propensity_for_t1),
np.mean(average_propensity_for_t2)))
print("Policy Risk: {} +- {}".format(np.mean(policy_risk),np.std(policy_risk)))
print("Ate_Error_0_1: {} +- {}".format(np.mean(test_ate_error_0_1),np.std(test_ate_error_0_1)))
print("Ate_Error_0_2: {} +- {}".format(np.mean(test_ate_error_0_2),np.std(test_ate_error_0_2)))
print("Ate_Error_1_2: {} +- {}".format(np.mean(test_ate_error_1_2),np.std(test_ate_error_1_2)))
print("Treatment accuracy: {} +- {}".format(np.mean(treatment_accuracy),np.std(treatment_accuracy)))
print("Treatment policy    :",treatment_policy)
print("Treatment prediction:",treatment_prediction)
print("Treatment label     :",treatment_label)
print("Factual accuracy of t0: {} +- {}".format(np.mean(test_factual_accuracy_of_t0),np.std(test_factual_accuracy_of_t0)))
print("Factual accuracy of t1: {} +- {}".format(np.mean(test_factual_accuracy_of_t1),np.std(test_factual_accuracy_of_t1)))
print("Factual accuracy of t2: {} +- {}".format(np.mean(test_factual_accuracy_of_t2),np.std(test_factual_accuracy_of_t2)))
print("Factual auc of t0: {} +- {}".format(np.mean(test_factual_auc_of_t0),np.std(test_factual_auc_of_t0)))
print("Factual auc of t1: {} +- {}".format(np.mean(test_factual_auc_of_t1),np.std(test_factual_auc_of_t1)))
print("Factual auc of t2: {} +- {}".format(np.mean(test_factual_auc_of_t2),np.std(test_factual_auc_of_t2)))
print("==========Average best train results=========")
print("average propensity for t0: {} and t1: {} and t2: {}".format(np.mean(train_average_propensity_for_t0),np.mean(train_average_propensity_for_t1),
np.mean(train_average_propensity_for_t2)))
print("Policy Risk: {} +- {}".format(np.mean(train_policy_risk),np.std(train_policy_risk)))
print("Ate_Error_0_1: {} +- {}".format(np.mean(train_ate_error_0_1),np.std(train_ate_error_0_1)))
print("Ate_Error_0_2: {} +- {}".format(np.mean(train_ate_error_0_2),np.std(train_ate_error_0_2)))
print("Ate_Error_1_2: {} +- {}".format(np.mean(train_ate_error_1_2),np.std(train_ate_error_1_2)))
print("Treatment accuracy: {} +- {}".format(np.mean(train_treatment_accuracy), np.std(train_treatment_accuracy)))
print("=============================================")
