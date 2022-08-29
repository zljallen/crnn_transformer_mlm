import os,torch
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
torch.cuda.device_count()

from utils import *
from DL_ClassifierModel import *
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
# import seaborn as sns

# dataClass = DataClass('../DataSet/', 'Data/train.txt', 'Data/test.txt', 'Data/vocabulary_agnostic.txt', 'Data/vocabulary_semantic.txt', hs=64, dropBLANK=True)
# dataClass = DataClass('../MiniData/', 'Data/Mtrain.txt', 'Data/Mtest.txt', 'Data/vocabulary_agnostic.txt', 'Data/vocabulary_semantic.txt', hs=64, dropBLANK=True)
# dataClass = DataClass('../DataSet/', 'Data/trainDel.txt', 'Data/testDel.txt', 'Data/vocabulary_agnostic.txt', 'Data/vocabulary_semantic.txt', hs=64, dropBLANK=True)
# with open('dataClass_dropBLANKDel.pkl', 'wb') as f:
#     pickle.dump(dataClass, f, 4)
with open('dataClass_dropBLANK.pkl', 'rb') as f:
     dataClass = pickle.load(f)
    
    
model = Trans2trans(classNum=len(dataClass.id2lab2), seqMaxLen=dataClass.maxItems*4+1+dataClass.maxItems+1,#dataClass.imgArr.shape[-1]+1+dataClass.maxItems+1, 
                    tknDropout=0.1, embDropout=0.0, hdnDropout=0.1, fcDropout=0.2,
                    imgHeight=64, contextSizeList=[1,5,25], feaSize=384, rnnLayerNum=2,
                    transNum=4, dk=48, multiNum=8, usePos=True, usePreLN=True,
                    maxItems=dataClass.maxItems, device=torch.device('cuda'))

# model = TransCTC(classNum=len(dataClass.id2lab2), seqMaxLen=max(dataClass.inputLen), imgHeight=64, contextSizeList=[1,9,81], 
#                  maxItems=dataClass.maxItems, cnnHiddenSize=128, 
#                  transNum=1, dk=32, multiNum=4, usePos=True,
#                  hdnDropout=0.1, device=torch.device('cuda'))

model.load('demo_79.pkl', map_location="cuda")
model.train(dataClass, trainSize=16, batchSize=16, validSize=128, epoch=500, stopRounds=-1, earlyStop=100, saveRounds=1, 
            optimType='Adam', preheat=0, lr1=0.0001, lr2=0.00001, weightDecay=0, isHigherBetter=False, metrics="editDistDivSeq", report=["seqErrorDivSeq", "editDistDivSeq", "editDistDivSymbol"], 
            savePath='demo')
