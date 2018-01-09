
# coding: utf-8

import pandas as pd
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import glob

for n_bins in range(8, 9): 
    
    # load data
    files = glob.glob("/Users/user/Dropbox/Coral Learning/img_data/*.txt")
    imgs = np.array([np.loadtxt(f) for f in files]).astype(np.uint8)

    db = np.genfromtxt("/Users/user/Dropbox/Coral Learning/coral_P8Dec17_v2.csv",delimiter=',',dtype=np.object)
    n_fish = np.array([int(f) for f in db[:,1]])
    fnames = db[:,0]
    
    lbls = db[:, n_bins].astype(np.uint8) 

    for n_sort in range(0, 3): 

        print("{}{} {}{}".format("NO.BINS:", n_bins, "RAND.SORT.ID:", n_sort))
        
        x = np.array(range(0,len(imgs)))
        np.random.shuffle(x)
        train_set = x[0:60]
        test_set = x[60:85]
        n_test = float(len(test_set))
        
        # define network
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=6, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer2 = nn.Sequential(
                    nn.Conv2d(8, 16, kernel_size=4, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.fc = nn.Linear(15376, n_bins)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.view(out.size(0), -1)

                out = self.fc(out)
                return out
            
        model = CNN()

        cr = torchvision.transforms.RandomCrop(124)
        tc = torchvision.transforms.CenterCrop(124)
        fl = torchvision.transforms.RandomHorizontalFlip()
        te = torchvision.transforms.ToTensor()

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        sfl = torch.nn.CrossEntropyLoss()

        for epoch in range(301):
            mb_size = 50
            mb_train = mb_size
            mb = torch.FloatTensor(mb_size,1,124,124)
            l = torch.LongTensor(mb_size,)

            model.train()

            train_lbs = dict(zip([str(label_i) for label_i in range(0,n_bins)], [0]*n_bins))
            idxs = np.random.choice(train_set, mb_size)
            for i,idx in enumerate(idxs):
                img = imgs[idx]
                img = Image.fromarray(img)

                img = cr(img)
                img = fl(img)
                img = te(img)
                mb[i] = img
                l[i] = lbls[idx].astype(np.int)
                train_lbs[str(l[i])] += 1

            mb, l = Variable(mb), Variable(l)

            optimizer.zero_grad()
            output = model(mb)

            loss = sfl(output,l)
            loss.backward()
            optimizer.step()

            if epoch%25==0:

                mb_size = 25
                mb = torch.FloatTensor(mb_size,1,124,124)
                l = torch.LongTensor(mb_size,)

                model.eval()

                test_lbs = []
                for i in range(mb_size):
                    idx = test_set[i]
                    img = imgs[idx]
                    img = Image.fromarray(img)

                    img = tc(img)
                    img = te(img)
                    mb[i] = img
                    l[i] = lbls[idx].astype(np.int)
                    test_lbs.append(l[i])

                mb, l = Variable(mb), l

                output = model(mb).data.numpy()
                pred = np.argmax(output,axis=1)
                gt = l.numpy()

                # compute accuracy of CNN's predictions
                a_cnn = np.sum(pred==gt)/n_test 

                # calculate the accuracy of a dumb classifier for this test set
                weighted_avg_test = 0.0
                for label in test_lbs:
                    prob_label = train_lbs[str(label)] / float(mb_train)
                    weighted_avg_test += prob_label 
                weighted_avg_test /= n_test

                # compare dumb clasifier accuracy to CNN acuraccy
                a_diff = a_cnn - weighted_avg_test
                print("{:01.3f} {:01.3f} {:01.3f}".format(weighted_avg_test, a_cnn, a_diff))
