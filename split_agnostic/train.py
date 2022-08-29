from utils import *
from DL_ClassifierModel import *
from matplotlib import pyplot as plt
import numpy as np
import os,torch

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #选择需要使用的GPU
torch.cuda.device_count()


# 读取数据
dataClass = DataClass_multi_output('../DataSet/', 'Data/train.txt', 'Data/test.txt', 'Data/vocabulary_agnostic.txt', hs=128, dropBLANK=True)


# 建立模型
model = Trans2trans_multi_output(   classNum1=dataClass.classNum1, classNum2=dataClass.classNum2, 
                                    labType2id=dataClass.labType2id, id2labType=dataClass.id2labType, labLoc2id=dataClass.labLoc2id, id2labLoc=dataClass.id2labLoc, 
                                    lab12id=dataClass.lab12id, id2lab1=dataClass.id2lab1,
                                    seqMaxLen=dataClass.maxItems*4+1+dataClass.maxItems+1,
                                    tknDropout=0.1, embDropout=0.0, hdnDropout=0.1, fcDropout=0.0,
                                    imgHeight=128, contextSizeList=[1,5,25,49], feaSize=1024, rnnLayerNum=2,
                                    transNum=4, dk=48, multiNum=8, usePos=True, usePreLN=True,
                                    maxItems=dataClass.maxItems, device=torch.device('cuda'))


model.train(dataClass, trainSize=16, batchSize=16, validSize=64, epoch=128, stopRounds=-1, earlyStop=20, saveRounds=1, 
            optimType='Adam', preheat=0, lr1=0.0001, lr2=0.01, weightDecay=0, isHigherBetter=False, metrics="editDistDivSeq", report=["seqErrorDivSeq", "editDistDivSeq", "editDistDivSymbol"], 
            savePath='TrainModel_splitAgnostic')