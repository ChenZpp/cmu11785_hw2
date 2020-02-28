import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataloader import *
from sklearn.metrics import roc_auc_score


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


def train(model, numEpochs, data_loader, test_loader, optimizer, criterion, device, task='Classification'):
    model.train()

    for epoch in range(numEpochs):

        avg_loss = 0.0
        start_time = time.time()

        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 100 == 99:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tTime: {:.1f}s'.format(epoch + 1, batch_num + 1,
                                                                                     avg_loss / 100,
                                                                                     time.time() - start_time))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        torch.save(model.state_dict(), './model' + str(epoch) + '.pt')
        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader, criterion, device)
            # train_loss, train_acc = test_classify(model, data_loader, criterion, device)
            print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(val_loss, val_acc))
            print("Time:", round(time.time() - start_time, 4))
            print('=' * 50)
        else:
            test_verify(model, test_loader)

    return model


def test_classify(model, test_loader, criterion, device):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total


def predict_classify(model, test_loader, device):
    model.eval()
    res = []

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        res += pred_labels.tolist()

    return res

def train_verify(model, numEpochs, data_loader, vari_loader, optimizer, criterion, device, task = 'verification'):
    model.train()

    for epoch in range(numEpochs):

        train_dataset = verification_train_dataset()
        data_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)

        avg_loss = 0.0
        start_time = time.time()

        for batch_num, (pos1, pos2, neg) in enumerate(data_loader):
            pos1, pos2, neg = pos1.to(device), pos2.to(device) , neg.to(device)

            optimizer.zero_grad()
            outputs1 = model(pos1)[0]
            outputs2 = model(pos2)[0]
            outputs3 = model(neg)[0]

            loss = criterion(outputs1, outputs2, outputs3)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 10 == 9:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tTime: {:.1f}s'.format(epoch + 1, batch_num + 1,
                                                                                     avg_loss / 10,
                                                                                     time.time() - start_time))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del pos1
            del pos2
            del neg
            del loss

        torch.save(model.state_dict(), './veri_model' + str(epoch) + '.pt')

        vari_verify(model, vari_loader, device)

    return model

def vari_verify(model, vari_loader, device):


    labels = []
    similarities = []
    for batch_num, (x1, x2, label) in enumerate(vari_loader):
        x1, x2 = x1.to(device), x2.to(device)

        outputs1 = model(x1)[0]
        outputs2 = model(x2)[0]

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(outputs1,outputs2).detach()
        print(similarity)
        labels += label.tolist()
        similarities += similarity

    print(roc_auc_score(labels, similarities))
    return roc_auc_score(labels, similarities)


