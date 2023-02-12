import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
import numpy as np
import torch


def collect_image(img_path):
    collect_image_data = np.zeros((3,224,224))
    get_img = sitk.ReadImage('../data/img/' + img_path[0]+'/Img_final_0.nii.gz')
    return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)
    collect_image_data = return_img[10:13]
    
    for num in range(1,len(img_path)):

        get_img = sitk.ReadImage('../data/img/' + img_path[num]+'/Img_final_0.nii.gz')
        return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)

        num_index = len(return_img) // 2
        collect_image_data = np.concatenate((collect_image_data, return_img[num_index-2:num_index+1]),0)
    return collect_image_data

class trainerData_collect(Dataset):
    def __init__(self, img_data, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_data = img_data
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        
        
        return_img = torch.from_numpy(self.img_data[index*3:index*3+3]).float().cuda()
        return return_data, return_yt, return_img
    def __len__(self):
        return len(self.data)     
        
class trainerData3d(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
           
        get_img = sitk.ReadImage('../data/img/' + self.img_path[index]+'/Img_final_0.nii.gz')
        return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)
   
        num_index = len(return_img) // 2
        return_img = return_img[num_index-10:num_index+10]
        return_img = return_img[np.newaxis,:,:,:]
        #return_img = return_img.repeat(3,axis=0)
        return_img = torch.from_numpy(return_img).float().cuda()
        return return_data, return_yt, return_img
    def __len__(self):
        return len(self.img_path)  

class trainerData3d_preload(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
        self.all_image_data = []
        for index in range(len(self.img_path)):
            
            get_img = sitk.ReadImage('../data/img/' + self.img_path[index]+'/Img_final_0.nii.gz')
            return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)
                
            num_index = len(return_img) // 2
            return_img = return_img[num_index-10:num_index+10]
            return_img = return_img[np.newaxis,:,:,:]
            self.all_image_data.append(return_img)
         
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()

        return_img = self.all_image_data[index]
        #return_img = return_img.repeat(3,axis=0)
        return_img = torch.from_numpy(return_img).float().cuda()
        return return_data, return_yt, return_img
    def __len__(self):
        return len(self.img_path)        



class trainerData_cli(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        #return_outcome = torch.from_numpy(self.outcome[index]).float().cuda()
        #return_treatment = torch.from_numpy(self.return_treatment[index]).float().cuda()
        
        
        return return_data, return_yt
    def __len__(self):
        return len(self.img_path)    


class trainerData(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        #return_outcome = torch.from_numpy(self.outcome[index]).float().cuda()
        #return_treatment = torch.from_numpy(self.return_treatment[index]).float().cuda()

        get_img = sitk.ReadImage('../data/img/' + self.img_path[index]+'/Img_final_0.nii.gz')
        return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)
            
        num_index = len(return_img) // 2
        return_img = torch.from_numpy(return_img[num_index - 2: num_index + 1]).float().cuda()
        return return_data, return_yt, return_img
    def __len__(self):
        return len(self.img_path)


class trainerData_single(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        #return_outcome = torch.from_numpy(self.outcome[index]).float().cuda()
        #return_treatment = torch.from_numpy(self.return_treatment[index]).float().cuda()
        
        return return_data, return_yt
    def __len__(self):
        return len(self.img_path)

def convert_file(x):
    x = x.values
    x = x.astype(float)
    return x


def load_and_format_covariates_hadcl(file_path):

    data = pd.read_excel(file_path)

    data = data.values[1:, ]

    #binfeats = list(range(6,37))
    #contfeats = [i for i in range(37) if i not in binfeats]

    mu_0, mu_1, path, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5], data[:, 6:]
    #perm = binfeats
    #x = x[:, perm].astype(float)

    return x.astype(float), path


def load_all_other_crap(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    t, y, y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    return t.reshape(-1, 1), y, y_cf, mu_0, mu_1

def load_all_other_crap_hadcl(file_path):
    data = pd.read_excel(file_path)
    data = data.values[1:, ]
    t, y, y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 6:]
    return t.reshape(-1, 1).astype(float), y.astype(float), y_cf.astype(float), mu_0.astype(float), mu_1.astype(float)

def main():
    pass


if __name__ == '__main__':
    main()
