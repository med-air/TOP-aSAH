from models import *
import os
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from SAH_data import *
import SimpleITK as sitk
import cv2
import numpy as np
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#torch.backends.cudnn.enabled = False
#random.shuffle
def policy_val(t, yf, q_t0, q_t1, q_t2, compute_policy_curve=False):

    # This function can caluculate the policy risk
    # Parameters
    # ----------
    # t : the factual treatment 
    # yf: the factual outcome
    # q_t0 : survival probability of treatment 0
    # q_t1 : survival probability of treatment 1
    # q_t2 : survival probability of treatment 2

    q_cat = np.concatenate((q_t0, q_t1),1)
    q_cat = np.concatenate((q_cat, q_t2),1)
    policy = np.argmax(q_cat,1)
    policy = policy[:,np.newaxis]
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

    # This function can caluculate the factual accuracy
    # Parameters
    # ----------
    # t : the factual treatment 
    # yf: the factual outcome
    # q_t0 : survival probability of treatment 0
    # q_t1 : survival probability of treatment 1
    # q_t2 : survival probability of treatment 2

    q_t0[q_t0>=0.5] = 1
    q_t0[q_t0<0.5] = 0
    
    q_t1[q_t1>=0.5] = 1
    q_t1[q_t1<0.5] = 0
    
    q_t2[q_t2>=0.5] = 1
    q_t2[q_t2<0.5] = 0
    
    accuracy_0 = np.sum(q_t0[t==0]==yf[t==0])/len(yf[t==0])
    accuracy_1 = np.sum(q_t1[t==1]==yf[t==1])/len(yf[t==1])
    accuracy_2 = np.sum(q_t2[t==2]==yf[t==2])/len(yf[t==2])
    
    print("Factual accuracy of t0:", accuracy_0)
    print("Factual accuracy of t1:", accuracy_1)
    print("Factual accuracy of t2:", accuracy_2)
    
def policy_risk_multi(t, yf, q_t0, q_t1, q_t2):
    policy_value = policy_val(t, yf, q_t0, q_t1, q_t2)
    policy_risk = 1 - policy_value
    return policy_risk


# Calculate the average treatment effect between treatment 0 and 1
def ate_error_0_1(t, yf, eff_pred):
    att = np.mean(yf[t==0]) - np.mean(yf[t==1])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)
    
# Calculate the average treatment effect between treatment 0 and 2
def ate_error_0_2(t, yf, eff_pred):
    att = np.mean(yf[t==0]) - np.mean(yf[t==2])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)

# Calculate the average treatment effect between treatment 1 and 2  
def ate_error_1_2(t, yf, eff_pred):
    att = np.mean(yf[t==1]) - np.mean(yf[t==2])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)

def _split_output(yt_hat, t, y, y_scaler, x, index, is_train=False):

        # Split output into dictionary for easier use in estimation
        # Args:
            # yt_hat: Generated prediction
            # t: Binary treatment assignments
            # y: Treatment outcomes
            # y_scaler: Scaled treatment outcomes
            # x: Covariates
            # index: Index in data

        # Returns:
            # Dictionary of all evaluation metrics

    traumatic = x[:,3]
    traumatic_index = np.where(traumatic==1)
    yt_hat = yt_hat[traumatic_index]
    t = t[traumatic_index]
    y = y[traumatic_index]
    y_scaler = y_scaler[traumatic_index]
    x = x[traumatic_index]
    
    yt_hat = yt_hat.detach().cpu().numpy()
    q_t0 = yt_hat[:, 0].reshape(-1, 1).copy()
    q_t1 = yt_hat[:, 1].reshape(-1, 1).copy()
    q_t2 = yt_hat[:, 2].reshape(-1, 1).copy()
    g = yt_hat[:, 6:9].copy()

    treatment_predicted = np.argmax(g,1)

    y = y.copy()
    var = "average propensity for t0: {} and t1: {} and t2: {}".format(g[:,0][t.squeeze() == 0.].mean(),
                                                                        g[:,1][t.squeeze() == 1.].mean(),g[:,2][t.squeeze() == 2.].mean())
    
    q_cat = np.concatenate((q_t0, q_t1),1)
    q_cat = np.concatenate((q_cat, q_t2),1)
    policy = np.argmax(q_cat,1)
    
    print(var)
    print("Policy Risk:", policy_risk_multi(t, y, q_t0, q_t1, q_t2))
    print("Ate_Error_0_1:", ate_error_0_1(t, y, q_t0 - q_t1))
    print("Ate_Error_0_2:", ate_error_0_2(t, y, q_t0 - q_t2))
    print("Ate_Error_1_2:", ate_error_1_2(t, y, q_t1 - q_t2))
    print("Treatment accuracy:", np.sum(treatment_predicted==t.squeeze())/treatment_predicted.shape[0])
    
    if not is_train:
        print("Treatment policy    :",policy)
        print("Treatment prediction:",treatment_predicted)
        print("Treatment label     :",t.squeeze().astype(int))
    
    factual_acc(t, y, q_t0, q_t1, q_t2)

    return {'ave propensity for t0': g[:,0][t.squeeze() == 0.].mean(), 'ave propensity for t1': g[:,1][t.squeeze() == 1.].mean(),
    'ave propensity for t2': g[:,2][t.squeeze() == 2.].mean(), 'Policy Risk': policy_risk_multi(t, y, q_t0, q_t1, q_t2), 
    'Ate_Error_0_1': ate_error_0_1(t, y, q_t0 - q_t1), 'Ate_Error_0_2': ate_error_0_2(t, y, q_t0 - q_t2),
    'Ate_Error_1_2': ate_error_1_2(t, y, q_t1 - q_t2), 'Treatment accuracy': np.sum(treatment_predicted==t.squeeze())/treatment_predicted.shape[0], 
    'Treatment policy': policy, 'Treatment prediction': treatment_predicted, 'Treatment label': t.squeeze().astype(int)}


def train(train_loader, net, optimizer, criterion, class_ratio):

    # Trains network for one epoch in batches.

    # Args:
    #     train_loader: Data loader for training set.
    #     net: Neural network model.
    #     optimizer: Optimizer (e.g. SGD).
    #     criterion: Loss function (e.g. cross-entropy loss).


    avg_loss = 0

    for i, data in enumerate(train_loader):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, images = data
        traumatic = inputs[:,3]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, sim_loss, close_loss = net(inputs,labels,images,traumatic)
        loss = criterion(outputs, labels, traumatic, class_ratio) + 1/9*sim_loss + 1/3*close_loss
        #loss = criterion(outputs, labels, traumatic, class_ratio)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)

def test(train_loader, net, criterion, number):

    # Trains network for one epoch in batches.

    # Args:
    #     train_loader: Data loader for training set.
    #     net: Neural network model.
    #     optimizer: Optimizer (e.g. SGD).
    #     criterion: Loss function (e.g. cross-entropy loss).

    net.eval()
    avg_loss = 0

    yt_hat_test = torch.from_numpy(np.zeros((number,9)))
    num_ = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, images = data
            traumatic = inputs[:,3]
            # zero the parameter gradients

            outputs = net(inputs,labels,images,traumatic, is_test=True)  
            yt_hat_test[num_:num_+outputs.shape[0]] = outputs
            num_ += outputs.shape[0]
    net.train()    
    return yt_hat_test

def load_image(path):
    get_test_X = sitk.ReadImage(path)
    test_X = sitk.GetArrayFromImage(get_test_X).astype(np.float32)
    image = np.zeros((test_X.shape[0],224,224)).astype(np.float32)
    for num in range(len(image)):
        image[num] = cv2.resize(test_X[num], (224, 224))
    return image

def train_and_predict_dragons(t, y, x, img_path, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross_3cls_ours, ratio=1., dragon='', val_split=0.2, batch_size=64, validation_index=0):

    # Method for training our proposed model and predicting new results
    # Returns:
    #     Outputs on train and test data


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    verbose = 0

    train_outputs_best = {}
    test_outputs_best = {}
    best_evaluation = 1.

    if dragon == 'tarnet':
        print('I am here making tarnet')
        net = TarNet(x.shape[1]).to("cuda")

    elif dragon == 'dragonnet':
        print("I am here making dragonnet")
        net = DragonNet(x.shape[1]).to("cuda")
        
    elif dragon == 'Ours':
        print("I am here Ours")
        net = MultiRL(x.shape[1]).to("cuda")

    # Which loss to use for training the network
    #net = torch.nn.DataParallel(net)
    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss
    
    # loss = knob_loss
    # for reporducing the IHDP experimemt

    i = 0
    torch.manual_seed(i)
    np.random.seed(i)
    # Get the data and optionally divide into train and test set

   # Sort out the training and validation data according to 10-fold cross validation
    all_index = np.arange(int(x.shape[0]))
    if validation_index == 9:
        test_index = np.arange(int(math.ceil(x.shape[0]/10)*validation_index),int(x.shape[0]))
    else:
        test_index = np.arange(int(math.ceil(x.shape[0]/10)*validation_index),int(math.ceil(x.shape[0]/10)*(validation_index+1)))
    train_index = []
    for m in all_index:
        if m not in test_index:
            train_index.append(m)
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]
    img_path_train, img_path_test = img_path[train_index], img_path[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)
    yt_test = np.concatenate([y_test, t_test], 1)

    t0_index = np.where(t_train==0)
    t1_index = np.where(t_train==1)
    t2_index = np.where(t_train==2)
    ratio_t0 = np.sum(y_train[t0_index])/len(y_train[t0_index])
    ratio_t1 = np.sum(y_train[t1_index])/len(y_train[t1_index])
    ratio_t2 = np.sum(y_train[t2_index])/len(y_train[t2_index])
    class_ratio = [ratio_t0, ratio_t1, ratio_t2]
    
    train_data = trainerData3d_preload(img_path_train, x_train, y_train, t_train, is_train = True)
    test_data = trainerData3d_preload(img_path_test, x_test, y_test, t_test, is_train = False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last = True)
    train_loader_test = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    import time;
    start_time = time.time()

    # Configuring optimizers
    # Training the networks first for 100 epochs with the Adam optimizer and
    # then for 300 epochs with the SGD optimizer.
    epochs1 = 1500
    epochs2 = 500

    # Add L2 regularization to t0 and t1 heads of the network
    
    # Set up the optimizer
    low_rate = 5e-4
    weight_decays = 5e-3
    optimizer_Adam = optim.Adam([{'params': net.representation_block.parameters(),'lr':low_rate},
                            {'params': net.t_predictions_cli.parameters()},
                            {'params': net.t_predictions_im.parameters()},
                            {'params': net.t0_head.parameters(),'weight_decay': weight_decays},
                            {'params': net.t1_head.parameters(),'weight_decay': weight_decays},
                            {'params': net.t2_head.parameters(),'weight_decay': weight_decays},
                            {'params': net.fc_im_0.parameters(),'weight_decay': weight_decays},
                            {'params': net.fc_im_1.parameters(),'weight_decay': weight_decays},
                            {'params': net.fc_im_2.parameters(),'weight_decay': weight_decays},
                            {'params': net.fc_cli_0.parameters(),'weight_decay': weight_decays},
                            {'params': net.fc_cli_1.parameters(),'weight_decay': weight_decays},
                            {'params': net.fc_cli_2.parameters(),'weight_decay': weight_decays},
                            {'params': net.share.parameters(),'lr':low_rate},
                            {'params': net.H_bar_im.parameters(),'lr':low_rate},
                            {'params': net.H_bar_cli.parameters(),'lr':low_rate},
                            {'params': net.pro_cli.parameters(),'lr':low_rate},
                            {'params': net.pro_im.parameters(),'lr':low_rate},
                            {'params': net.resenet_head.parameters(),'lr':low_rate}], lr=5e-3)
    scheduler_Adam = optim.lr_scheduler.StepLR(optimizer=optimizer_Adam, step_size = 300, gamma=0.5)       
    
 

    train_loss = 0
    epochs0 = 0
    
    if epochs0 != 0:
        load_model_path = '../models_save/ours/'+str(epochs0)+'.pth'
        net.load_state_dict(torch.load(load_model_path))

    
    for epoch in range(epochs0, epochs1):
        # Adam training run
        train_loss = train(train_loader, net, optimizer_Adam, loss, class_ratio)
        scheduler_Adam.step(train_loss)
        
        # Evaluate the model on the validation set and save the results every 10 epoch
        if epoch % 10 ==0:
            print(str(epoch)+"/"+str(epochs1)+" "+f"Adam loss: {train_loss}")
            yt_hat_test = test(test_loader, net, loss, len(test_index))
            yt_hat_train = test(train_loader_test, net, loss, len(train_index))
            np.savez_compressed("../results_save/ours_woDB/{}_fold_{}_epoch_test.npz".format(validation_index, epoch),yt_hat_test=yt_hat_test,t_test=t_test,y_test=y_test,
            y=y,x_test=x_test)
            np.savez_compressed("../results_save/ours_woDB/{}_fold_{}_epoch_train.npz".format(validation_index, epoch),yt_hat_train=yt_hat_train,t_train=t_train,y_train=y_train,
            y=y,x_train=x_train)
            test_outputs = _split_output(yt_hat_test, t_test, y_test, y, x_test, test_index, is_train=False)
            train_outputs = _split_output(yt_hat_train, t_train, y_train, y, x_train, train_index, is_train=True)
            if test_outputs['Policy Risk'] <= best_evaluation:
                train_outputs_best = train_outputs
                test_outputs_best = test_outputs
                best_evaluation = test_outputs['Policy Risk']
            print("==================the {} fold====================".format(validation_index))

        # Save the model every 100 epoch
        if epoch % 100 ==0:
            save_model_path = '../models_save/ours_woDB/'+str(epoch)+'.pth'
            torch.save(net.state_dict(),save_model_path)
    save_model_path = '../models_save/ours_woDB/'+str(epoch)+ '_' + str(validation_index) + '_fold.pth'
    torch.save(net.state_dict(),save_model_path)       
    return test_outputs_best, train_outputs_best


def run_ihdp(data_base_dir='./data/SAH', output_dir='~/result/ihdp/',
             knob_loss=dragonnet_loss_binarycross_3cls_ours,
             ratio=1., dragon=''):

    print("the model is {}".format(dragon))

    simulation_files = sorted(glob.glob("{}/*.xls".format(data_base_dir)))

    for idx, simulation_file in enumerate(simulation_files):

        simulation_output_dir = os.path.join(output_dir, str(idx))

        os.makedirs(simulation_output_dir, exist_ok=True)

        x, img_path = load_and_format_covariates_hadcl(simulation_file)
        t, y, y_cf, mu_0, mu_1 = load_all_other_crap_hadcl(simulation_file)
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)

        average_propensity_for_t0 = []
        average_propensity_for_t1 = []
        average_propensity_for_t2 = []
        policy_risk = []
        ate_error_0_1 = []
        ate_error_0_2 = []
        ate_error_1_2 = []
        treatment_accuracy = []
        treatment_policy=np.array([])
        treatment_prediction=np.array([])
        treatment_label=np.array([])

        
        train_average_propensity_for_t0 = []
        train_average_propensity_for_t1 = []
        train_average_propensity_for_t2 = []
        train_policy_risk = []
        train_ate_error_0_1 = []
        train_ate_error_0_2 = []
        train_ate_error_1_2 = []
        train_treatment_accuracy = []
       
        # Perform 10-fold cross validation
        for validation_index in range(0,10):
            # print("Is targeted regularization: {}".format(is_targeted_regularization))
            test_outputs_best, train_outputs_best = train_and_predict_dragons(t, y, x, img_path,
                                                                   targeted_regularization=False,
                                                                   output_dir=simulation_output_dir,
                                                                   knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                                   val_split=0.2, batch_size=256, validation_index=validation_index)
                  
            # Evaluate the model after each cross validation
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
            print("==========Best train results for the {} fold==========".format(validation_index))
            print("average propensity for t0: {} and t1: {} and t2: {}".format(train_outputs_best['ave propensity for t0'],train_outputs_best['ave propensity for t1'],
            train_outputs_best['ave propensity for t2']))
            print("Policy Risk:", train_outputs_best['Policy Risk'])
            print("Ate_Error_0_1:", train_outputs_best['Ate_Error_0_1'])
            print("Ate_Error_0_2:", train_outputs_best['Ate_Error_0_2'])
            print("Ate_Error_1_2:", train_outputs_best['Ate_Error_1_2'])
            print("Treatment accuracy:", train_outputs_best['Treatment accuracy'])
            print("====================================================")
            average_propensity_for_t0.append(test_outputs_best['ave propensity for t0'])
            average_propensity_for_t1.append(test_outputs_best['ave propensity for t1'])
            average_propensity_for_t2.append(test_outputs_best['ave propensity for t2'])
            policy_risk.append(test_outputs_best['Policy Risk'])
            ate_error_0_1.append(test_outputs_best['Ate_Error_0_1'])
            ate_error_0_2.append(test_outputs_best['Ate_Error_0_2'])
            ate_error_1_2.append(test_outputs_best['Ate_Error_1_2'])
            treatment_accuracy.append(test_outputs_best['Treatment accuracy'])
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
            train_treatment_accuracy.append(train_outputs_best['Treatment accuracy'])
                   
        print("==========Average best test results==========")
        print("average propensity for t0: {} and t1: {} and t2: {}".format(np.mean(average_propensity_for_t0),np.mean(average_propensity_for_t1),
        np.mean(average_propensity_for_t2)))
        print("Policy Risk:", np.mean(policy_risk))
        print("Ate_Error_0_1:", np.mean(ate_error_0_1))
        print("Ate_Error_0_2:", np.mean(ate_error_0_2))
        print("Ate_Error_1_2:", np.mean(ate_error_1_2))
        print("Treatment accuracy:", np.mean(treatment_accuracy))
        print("Treatment policy    :",treatment_policy)
        print("Treatment prediction:",treatment_prediction)
        print("Treatment label     :",treatment_label)
        print("==========Average best train results=========")
        print("average propensity for t0: {} and t1: {} and t2: {}".format(np.mean(train_average_propensity_for_t0),np.mean(train_average_propensity_for_t1),
        np.mean(train_average_propensity_for_t2)))
        print("Policy Risk:", np.mean(train_policy_risk))
        print("Ate_Error_0_1:", np.mean(train_ate_error_0_1))
        print("Ate_Error_0_2:", np.mean(train_ate_error_0_2))
        print("Ate_Error_1_2:", np.mean(train_ate_error_1_2))
        print("Treatment accuracy:", np.mean(train_treatment_accuracy))
        print("=============================================")


def turn_knob(data_base_dir='./data/SAH', knob='dragonnet',
              output_base_dir=''):
    output_dir = os.path.join(output_base_dir, knob)

    if knob == 'dragonnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='dragonnet')

    if knob == 'tarnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='tarnet')

    if knob == 'Ours':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='Ours')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD",default='./data/SAH')
    parser.add_argument('--knob', type=str, default='Ours',
                        help="dragonnet or tarnet or Ours")

    parser.add_argument('--output_base_dir', type=str, help="directory to save the output",default='../result/hadcl_single')

    args = parser.parse_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir)


if __name__ == '__main__':
    main()
