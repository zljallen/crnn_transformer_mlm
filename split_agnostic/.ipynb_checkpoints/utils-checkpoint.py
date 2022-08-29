import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms as transforms
from collections import Counter
import torch,random

class DataClass_multi_output:
    def __init__(self, rootPath, trainPath, testPath, label1Path, validSize=0.1, hs=64, dropBLANK=False):        
        # Load file
        print('Loading files...')
        with open(trainPath, 'r') as f:
            trainNames = [i.strip() for i in f.readlines()]
        with open(testPath, 'r') as f:
            testNames = [i.strip() for i in f.readlines()]
        lab12id,id2lab1 = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        labType2id,id2labType = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        labLoc2id,id2labLoc = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        cnt1,cnt2 = 3,3
        with open(label1Path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                char = line.strip()
                lab12id[char] = i+3
                id2lab1.append(char)
                
                idx = char.find('-')
                p1,p2 = char[:idx],char[idx+1:]
                
                if p1 not in labType2id:
                    labType2id[p1] = cnt1
                    id2labType.append(p1)
                    cnt1 += 1
                if p2 not in labLoc2id:
                    labLoc2id[p2] = cnt2
                    id2labLoc.append(p2)
                    cnt2 += 1
        # Open image
        print('Openning images...')
        imgs,labTypes,labLocs = [],[],[]
        lab1s = []
        wMax = -1
        for name in tqdm(trainNames+testNames):
            path = os.path.join(rootPath, name)
            img = Image.open(os.path.join(path, f"{name}.png"))
            if dropBLANK:
                arr = np.array(img)
                ctr = arr.sum(axis=0)
                blank = Counter(ctr).most_common(1)[0][0]
                arr = arr[:,ctr>blank]
                img = Image.fromarray(arr)
            w,h = img.size
            w,h = int(w*hs/h),hs
            if w>wMax:
                wMax = w
            imgs.append( img.resize((w,h)) )
            with open(os.path.join(path, f"{name}.agnostic"), 'r') as f:
                tmp = f.readline().split()
                lab1s.append( tmp )
                
                p1s,p2s = [],[]
                for char in tmp:
                    idx = char.find('-')
                    p1,p2 = char[:idx],char[idx+1:]
                    p1s.append(p1)
                    p2s.append(p2)
                
                labTypes.append(p1s)
                labLocs.append(p2s)
        trainIdList = list(range(len(trainNames)))
        print('Converting images...')
        imgArr = []
        for i in tqdm(imgs): 
            imgArr.append( np.hstack([np.array(i),np.zeros((hs,wMax-i.size[0]))]).astype('bool') )
        self.imgArr = np.array(imgArr, dtype='bool')
        print('Finishing...')
        self.trainIdList, self.validIdList = train_test_split(trainIdList, test_size=validSize)
        self.testIdList = list(range(len(trainNames), len(imgs)))
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(self.trainIdList),len(self.validIdList),len(self.testIdList)
        self.imgs = imgs
        self.lab1s = lab1s
        self.labTypes = labTypes
        self.labLocs = labLocs
        self.lab12id,self.id2lab1 = lab12id,id2lab1
        self.labType2id,self.id2labType = labType2id,id2labType
        self.labLoc2id,self.id2labLoc = labLoc2id,id2labLoc
        self.classNum1,self.classNum2 = len(id2labType),len(id2labLoc)
        self.inputLen = np.array([i.size[0] for i in imgs], dtype='int32')

        maxItems = np.max([len(i) for i in labTypes])
        self.maxItems = maxItems
        self.labArr = np.array([[2]+[self.lab12id[i] for i in labs]+[1]+[0]*(maxItems-len(labs)) for labs in lab1s], dtype=np.int32)
        self.labArr_multi_output = np.array([[[2,2]]+[[self.labType2id[i1], self.labLoc2id[i2]] for i1,i2 in zip(p1s,p2s)]+[[1,1]]+[[0,0]]*(maxItems-len(p1s)) for p1s,p2s in zip(labTypes,labLocs)], dtype=np.int32)
        self.targetLen = np.array([len(i) for i in labTypes], dtype='int32')
        
    def describe(self):
        pass

    def random_batch_data_stream(self, batchSize=64, type='train', device=torch.device('cpu')):
        idList = list(self.trainIdList) if type=='train' else list(self.validIdList)
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                yield {
                    "imgArr": torch.tensor(self.imgArr[samples]-0.5, dtype=torch.float32).to(device),
                    "imgLab": torch.tensor(self.labArr[samples], dtype=torch.long).to(device),
                    "imgLab_multi_output": torch.tensor(self.labArr_multi_output[samples], dtype=torch.long).to(device),
                    "inputLen": torch.tensor(self.inputLen[samples], dtype=torch.long).to(device),
                    "targetLen": torch.tensor(self.targetLen[samples], dtype=torch.long).to(device), 
                }, {
                    "imgLab":torch.tensor(self.labArr[samples][:,1:], dtype=torch.long).to(device),
                    "imgLab_multi_output":torch.tensor(self.labArr_multi_output[samples][:,1:], dtype=torch.long).to(device)
                }

    def one_epoch_batch_data_stream(self, batchSize=64, type='valid', device=torch.device('cpu')):
        if type=='train':
            idList = list(self.trainIdList)
        elif type=='valid':
            idList = list(self.validIdList)
        else:
            idList = list(self.testIdList)
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            yield {
                "imgArr": torch.tensor(self.imgArr[samples]-0.5, dtype=torch.float32).to(device),
                "imgLab": torch.tensor(self.labArr[samples], dtype=torch.long).to(device),
                "imgLab_multi_output": torch.tensor(self.labArr_multi_output[samples], dtype=torch.long).to(device),
                "inputLen": torch.tensor(self.inputLen[samples], dtype=torch.long).to(device),
                "targetLen": torch.tensor(self.targetLen[samples], dtype=torch.long).to(device), 
            }, {
                "imgLab":torch.tensor(self.labArr[samples][:,1:], dtype=torch.long).to(device),
                "imgLab_multi_output":torch.tensor(self.labArr_multi_output[samples][:,1:], dtype=torch.long).to(device)
            }
