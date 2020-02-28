import os
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        if target_list != None:
            self.n_class = len(list(set(target_list)))
        else:
            self.n_class = -1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        if self.target_list != None:
            label = self.target_list[index]
        else:
            label = -1
        return img, label


def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n


def load_classification_train(folder_root = r'validation_classification', medium = True,  batch_size = 10, shuffle = True, num_workers = 1):
    root = folder_root + r'/medium' if medium else r'/large'
    imageFolder_dataset = torchvision.datasets.ImageFolder(root = root,
                                                           transform=torchvision.transforms.ToTensor())
    return DataLoader(imageFolder_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),imageFolder_dataset

def load_classification_test(root = r'test_classification/medium', batch_size = 10, num_workers = 1):
    img_list = []

    order = open("test_order_classification.txt", "r")
    file_names = order.read().split()
    for file_name in file_names:
        img_list.append(os.path.join(root,file_name))

    #print(img_list)

    testset =  ImageDataset(img_list, None)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),testset

#def load_verification_train(folder_root = r'test_classification/medium', batch_size = 10, num_workers = 1):

class verification_train_dataset(Dataset):
    def __init__(self, path = r'validation_classification/medium'):
        self.path = path
        num_id = len(os.listdir(path))
        self.pos_folder = os.listdir(path)
        self.neg_folder = [ str((int(i) + np.random.randint(1, num_id - 1)) % num_id) for i in self.pos_folder]

    def __len__(self):
        return len(self.pos_folder)

    def __getitem__(self, index):
        pos0,pos1 = np.random.choice(os.listdir(self.path + r"/" + self.pos_folder[index]),2)
        neg = np.random.choice(os.listdir(self.path + r"/" + self.neg_folder[index]))
        img_pos0 = Image.open(self.path + r"/" + self.pos_folder[index] + r'/' +  pos0)
        img_pos1 = Image.open(self.path + r"/" + self.pos_folder[index] + r'/' + pos1)
        img_neg = Image.open(self.path + r"/" + self.neg_folder[index] + r'/' + neg)
        return torchvision.transforms.ToTensor()(img_pos0),torchvision.transforms.ToTensor()(img_pos1),torchvision.transforms.ToTensor()(img_neg)

class verification_validation_dataset(Dataset):
    def __init__(self, path = r'validation_verification'):
        self.namelist = open('validation_trials_verification.txt','r').read().split()
        self.path = path

    def __len__(self):
        return int(len(self.namelist)/3)

    def __getitem__(self, index):
        img1 = Image.open(os.path.join(self.path, self.namelist[index*3]))
        img2 = Image.open(self.path + r'/' + self.namelist[index*3+1])
        same = int(self.namelist[index*3+2])
        return torchvision.transforms.ToTensor()(img1), torchvision.transforms.ToTensor()(img2), same

class verification_test_dataset(Dataset):
    def __init__(self, path = r'test_verification'):
        self.namelist = open(r'test_trials_verification_student.txt','r').read().split()
        self.path = path

    def __len__(self):
        return int(len(self.namelist))

    def __getitem__(self, index):
        img1 = Image.open(self.path + r'/' + self.namelist[index * 2])
        img2 = Image.open(self.path + r'/' + self.namelist[index * 2 + 1])
        return torchvision.transforms.ToTensor()(img1), torchvision.transforms.ToTensor()(img2)






