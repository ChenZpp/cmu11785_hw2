import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from dataloader import *
from train import *
from models import *

# data loader
train_dataloader, train_dataset = load_classification_train(folder_root = r'validation_classification', medium = True,  batch_size = 32, shuffle = True, num_workers = 1)
valid_dataloader, valid_dataset = load_classification_train(folder_root = r'validation_classification', medium = True,  batch_size = 10, shuffle = False, num_workers = 1)
test_dataloader, test_dataset = load_classification_test()

# parameters
numEpochs = 2
num_feats = 3

learningRate = 1e-2
weightDecay = 5e-5

hidden_sizes = [2,2,2]
num_classes = 2300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using cuda' if torch.cuda.is_available() else 'using cpu')

#network = Network(num_feats, hidden_sizes, num_classes)
network = Resnet(num_feats,hidden_sizes,num_classes)
#network.load_state_dict(torch.load("model7.pt"))
print(network)
network.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)

# train
network.train()
network.to(device)
model = train(network, numEpochs, train_dataloader, valid_dataloader, optimizer, criterion, device)

# predict
result = predict_classify(model, test_dataloader, device)
for i in range(len(result)):
    result[i] = train_dataset.classes[result[i]]
result = pd.DataFrame({'id': range(5000, 5000+len(result)), 'label': result})
result.to_csv("result.csv",index=False)

torch.save(model.state_dict(),'./model.pt')
