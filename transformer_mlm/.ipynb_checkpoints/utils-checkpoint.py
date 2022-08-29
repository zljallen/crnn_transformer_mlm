import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms as transforms
from collections import Counter
import torch,random

class DataClass_Batch:
    def __init__(self, rootPath, trainPath, testPath, label1Path, label2Path, encoding='agnostic', validSize=0.1, hs=128, dropBLANK=False):  
        # Load 图像file
        print('Loading files...')
        with open(trainPath, 'r') as f:
            trainNames = [i.strip() for i in f.readlines()]
        with open(testPath, 'r') as f:
            testNames = [i.strip() for i in f.readlines()]
        # Agnostic标签数字化
        lab12id,id2lab1 = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        with open(label1Path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                char = line.strip()
                lab12id[char] = i+3
                id2lab1.append(char)
        # Semantic标签数字化
        lab22id,id2lab2 = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        with open(label2Path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                char = line.strip()
                lab22id[char] = i+3
                id2lab2.append(char)
        # Open image
        print('Openning images...')
        imgArr = []
        imgs,lab1s,lab2s = [],[],[]
        widths = []
        
        for name in tqdm((trainNames+testNames)):
            path = os.path.join(rootPath, name)
            img = Image.open(os.path.join(path, f"{name}.png"))
            
            if dropBLANK:
                arr = np.array(img)
                ctr = arr.sum(axis=0)
                blank = Counter(ctr).most_common(1)[0][0]
                arr = arr[:,ctr>blank]
                img = Image.fromarray(arr)
                ##############
                imgs.append(img)
            w,h = img.size
            w,h = int(w*hs/h),hs
            widths.append(w)
#             if w>wMax:
#                 wMax = w
            img = img.resize((w,h))
#             imgs.append(img)
            imgArr.append( list(np.array(img, dtype='bool')) )  #img: (h,w)
            with open(os.path.join(path, f"{name}.agnostic"), 'r') as f:
                lab1s.append( f.readline().split() )
            with open(os.path.join(path, f"{name}.semantic"), 'r') as f:
                lab2s.append( f.readline().split() )
        
        trainIdList = list(range(len(trainNames)))
        print('Creating indexes...')
        self.imgArr = np.array(imgArr)
        print('Finishing...')
        self.trainIdList, self.validIdList = train_test_split(trainIdList, test_size=validSize)
        self.testIdList = list(range(len(trainNames), len(imgArr)))
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(self.trainIdList),len(self.validIdList),len(self.testIdList)
        self.imgs = imgs
        self.widths = np.array(widths, dtype=np.int32)
        self.lab1s = lab1s
        self.lab2s = lab2s
        self.lab12id,self.id2lab1 = lab12id,id2lab1
        self.lab22id,self.id2lab2 = lab22id,id2lab2
        self.classNum1,self.classNum2 = len(id2lab1),len(id2lab2)
        self.inputLen = np.array([len(i[0]) for i in imgArr], dtype='int32') #获取所有输入图像的实际宽
        
        # Agnostic编码
        if encoding=='agnostic':
            maxItems = np.max([len(i) for i in lab1s])
            self.maxItems = maxItems
            self.labArr = np.array([[2]+[self.lab12id[i] for i in labs]+[1]+[0]*(maxItems-len(labs)) for labs in lab1s], dtype=np.int32)
            self.targetLen = np.array([len(i) for i in lab1s], dtype='int32')
            self.label = id2lab1

        # Semantic编码
        if encoding=='semantic':
            maxItems = np.max([len(i) for i in lab2s])
            self.maxItems = maxItems
            self.labArr = np.array([[2]+[self.lab22id[i] for i in labs]+[1]+[0]*(maxItems-len(labs)) for labs in lab2s], dtype=np.int32)
            self.targetLen = np.array([len(i) for i in lab2s], dtype='int32')#所有输入图像相应标签的实际长度
            self.label = id2lab2

    def describe(self):
        pass

    def random_batch_data_stream(self, batchSize=64, type='train', device=torch.device('cpu')):
        idList = list(self.trainIdList) if type=='train' else list(self.validIdList)
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
#                 print('Converting images...')
                imgs = self.imgArr[samples]
                wMax = self.widths.max()
                imgArr = [] 
                for j in imgs:
                    imgArr.append( np.hstack([np.vstack(j),np.zeros((len(j),wMax-len(j[0])), dtype='bool')]).astype('bool') )
                imgArr = np.array(imgArr, dtype='bool')
#                 imgArr = np.array([np.hstack([np.vstack(j),np.zeros((len(j),wMax-len(j[0])), dtype='bool')]) for j in imgs], dtype='bool')
            
                yield {
                    "imgArr": torch.tensor(imgArr-0.5, dtype=torch.float32).to(device),
                    "imgLab": torch.tensor(self.labArr[samples], dtype=torch.long).to(device),
                    "inputLen": torch.tensor(self.inputLen[samples], dtype=torch.long).to(device),
                    "targetLen": torch.tensor(self.targetLen[samples], dtype=torch.long).to(device), 
                }, torch.tensor(self.labArr[samples][:,1:], dtype=torch.long).to(device)

    def one_epoch_batch_data_stream(self, batchSize=64, type='valid', device=torch.device('cpu')):
        if type=='train':
            idList = list(self.trainIdList)
        elif type=='valid':
            idList = list(self.validIdList)
        else:
            idList = list(self.testIdList)
        
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            imgs = self.imgArr[samples]
            wMax = self.widths.max()
            imgArr = []       
            for j in imgs:
                imgArr.append( np.hstack([np.vstack(j),np.zeros((len(j),wMax-len(j[0])))]).astype('bool') )
            imgArr = np.array(imgArr, dtype='bool')
#             imgArr = np.array([np.hstack([np.vstack(j),np.zeros((len(j),wMax-len(j[0])), dtype='bool')]) for j in imgs], dtype='bool')
            yield {
                "imgArr": torch.tensor(imgArr-0.5, dtype=torch.float32).to(device),
                "imgLab": torch.tensor(self.labArr[samples], dtype=torch.long).to(device),
                "inputLen": torch.tensor(self.inputLen[samples], dtype=torch.long).to(device),
                "targetLen": torch.tensor(self.targetLen[samples], dtype=torch.long).to(device), 
            }, torch.tensor(self.labArr[samples][:,1:], dtype=torch.long).to(device)

class DataClass:
    def __init__(self, rootPath, trainPath, testPath, label1Path, label2Path, encoding='agnostic', validSize=0.1, hs=64, dropBLANK=False):        
        # Load 图像file
        print('Loading files...')
        with open(trainPath, 'r') as f:
            trainNames = [i.strip() for i in f.readlines()]
        with open(testPath, 'r') as f:
            testNames = [i.strip() for i in f.readlines()]
        # Agnostic标签数字化
        lab12id,id2lab1 = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        with open(label1Path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                char = line.strip()
                lab12id[char] = i+3
                id2lab1.append(char)
        # Semantic标签数字化
        lab22id,id2lab2 = {"<PAD>":0,"<EOS>":1,"<SOS>":2},["<PAD>","<EOS>","<SOS>"]
        with open(label2Path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                char = line.strip()
                lab22id[char] = i+3
                id2lab2.append(char)
        # Open image
        print('Openning images...')
        imgs,lab1s,lab2s = [],[],[]
        wMax = -1
        for name in tqdm(trainNames+testNames):
            path = os.path.join(rootPath, name)
#             with open(os.path.join(path, f"{name}.png"), 'rb') as fi:
            img = Image.open(os.path.join(path, f"{name}.png"))
            if dropBLANK:
                arr = np.array(img)
                ctr = arr.sum(axis=0)
                blank = Counter(ctr).most_common(1)[0][0]
                arr = arr[:,ctr>blank]
                img = Image.fromarray(arr)
            w,h = img.size
#             print(w,h)
            w,h = int(w*hs/h),hs
#             print(w,h)
            if w>wMax:
                wMax = w
            imgs.append( img.resize((w,h)) )
            with open(os.path.join(path, f"{name}.agnostic"), 'r') as f:
                lab1s.append( f.readline().split() )
            with open(os.path.join(path, f"{name}.semantic"), 'r') as f:
                lab2s.append( f.readline().split() )
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
        self.lab2s = lab2s
        self.lab12id,self.id2lab1 = lab12id,id2lab1
        self.lab22id,self.id2lab2 = lab22id,id2lab2
        self.classNum1,self.classNum2 = len(id2lab1),len(id2lab2)
        self.inputLen = np.array([i.size[0] for i in imgs], dtype='int32') #获取所有输入图像的实际宽
        
        # Agnostic编码
        if encoding=='agnostic':
            maxItems = np.max([len(i) for i in lab1s])
            self.maxItems = maxItems
            self.labArr = np.array([[2]+[self.lab12id[i] for i in labs]+[1]+[0]*(maxItems-len(labs)) for labs in lab1s], dtype=np.int32)
            self.targetLen = np.array([len(i) for i in lab1s], dtype='int32')
        
        # Semantic编码
        if encoding=='semantic':
            maxItems = np.max([len(i) for i in lab2s])
            self.maxItems = maxItems
            self.labArr = np.array([[2]+[self.lab22id[i] for i in labs]+[1]+[0]*(maxItems-len(labs)) for labs in lab2s], dtype=np.int32)
            self.targetLen = np.array([len(i) for i in lab2s], dtype='int32')#所有输入图像相应标签的实际长度
        
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
                    "inputLen": torch.tensor(self.inputLen[samples], dtype=torch.long).to(device),
                    "targetLen": torch.tensor(self.targetLen[samples], dtype=torch.long).to(device), 
                }, torch.tensor(self.labArr[samples][:,1:], dtype=torch.long).to(device)

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
                "inputLen": torch.tensor(self.inputLen[samples], dtype=torch.long).to(device),
                "targetLen": torch.tensor(self.targetLen[samples], dtype=torch.long).to(device), 
            }, torch.tensor(self.labArr[samples][:,1:], dtype=torch.long).to(device)
