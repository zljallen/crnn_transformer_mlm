from utils import *
import os,torch,math
from DL_ClassifierModel import *
from matplotlib import pyplot as plt
import numpy as np
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #选择需要使用的GPU
torch.cuda.device_count()

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-encoding', dest='encoding', type=str, required=True, help='Encoding type (semantic or agnostic).')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
args = parser.parse_args()

# 直接读取实际数据
dataClass = DataClass_Batch('../DataSet', 'Data/train.txt', 'Data/test.txt', 'Data/vocabulary_agnostic.txt', 'Data/vocabulary_semantic.txt', encoding=args.encoding, hs=128, dropBLANK=True)

# 建立模型
model = Trans2trans(classNum=len(dataClass.label), seqMaxLen=dataClass.maxItems*4+1+dataClass.maxItems+1,
                    tknDropout=0.1, embDropout=0.0, hdnDropout=0.1, fcDropout=0.0,
                    imgHeight=128, contextSizeList=[1,5,25,49], feaSize=1024, rnnLayerNum=2,
                    transNum=4, dk=48, multiNum=8, usePos=True, usePreLN=True,
                    maxItems=dataClass.maxItems, device=torch.device('cuda'))


#模型训练并保存最好的模型
model.train(dataClass, trainSize=16, batchSize=16, validSize=64, epoch=128, stopRounds=-1, earlyStop=20, saveRounds=1, 
            optimType='Adam', lr=0.0001, weightDecay=0, warmupEpochs=0, isHigherBetter=False, metrics="editDistDivSeq", report=["seqErrorDivSeq", "editDistDivSeq", "editDistDivSymbol"], 
            savePath=args.save_model, metrics_savefile="tmp.npz")