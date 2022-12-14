from utils import *
from DL_ClassifierModel import *
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import cv2
import os,torch

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #选择需要使用的GPU
torch.cuda.device_count()

# 读取数据
dataClass = DataClass_multi_output('../DataSet/', 'Data/train.txt', 'Data/test.txt', 'Data/vocabulary_agnostic.txt', hs=128, dropBLANK=True)


#建立模型
model = Trans2trans_multi_output(   classNum1=dataClass.classNum1, classNum2=dataClass.classNum2, 
                                    labType2id=dataClass.labType2id, id2labType=dataClass.id2labType, labLoc2id=dataClass.labLoc2id,                                                     id2labLoc=dataClass.id2labLoc, 
                                    lab12id=dataClass.lab12id, id2lab1=dataClass.id2lab1,
                                    seqMaxLen=dataClass.maxItems*4+1+dataClass.maxItems+1,
                                    tknDropout=0.1, embDropout=0.0, hdnDropout=0.1, fcDropout=0.0,
                                    imgHeight=128, contextSizeList=[1,5,25,49], feaSize=1024, rnnLayerNum=2,
                                    transNum=4, dk=48, multiNum=8, usePos=True, usePreLN=True,
                                    maxItems=dataClass.maxItems, device=torch.device('cuda'))


#加载已训练好的模型
model.load('TrainModel_splitAgnostic.pkl', map_location="cuda")
model.to_eval_mode()

#选取一批数据
def batch_image_predict(batchImgArr):
    bs = len(batchImgArr)
    imgArr = torch.tensor(batchImgArr, dtype=torch.float32, device=model.device).transpose(1,2)
    imgLab = torch.cat([torch.tensor([[[2,2]]]*bs,device=model.device),torch.zeros((bs,model.maxItems+1,2), dtype=torch.long, device=model.device)], dim=1)
    imgLab_type = torch.cat([torch.tensor([[2]],device=model.device),torch.zeros((1,model.maxItems+1), dtype=torch.long, device=model.device)], dim=1)
    imgLab_loc = torch.cat([torch.tensor([[2]],device=model.device),torch.zeros((1,model.maxItems+1), dtype=torch.long, device=model.device)], dim=1)
    cnt = 1
    x = torch.cat(model.seqCNN(imgArr), dim=-1)
    x_pool = F.adaptive_max_pool1d(x.transpose(1,2), model.maxItems*4).transpose(1,2)
    while True:
        imgLab_type = imgLab[:,:,0]
        imgLab_loc = imgLab[:,:,1]
        
        y1 = model.symEmbedding1(imgLab_type)
        y2 = model.symEmbedding2(imgLab_loc)
        y = y1 + y2
        # x_pool: 1 × seqLen × feaSize
        B,L,C = x_pool.shape
        x_rnn, hn = model.eSeqRNN(torch.cat([x_pool,y[:,:1]], dim=1)) # => batchSize × seqLen × hiddenSize*2
        eos = x_rnn[:,-1:]
        x_rnn = x_rnn[:,:-1]
        x = x_rnn + x_pool
        x = torch.cat([x, eos], dim=1)
        if type(hn)==tuple:
            hn,cn = hn
            hn = hn.view(model.rnnLayerNum, 2, B, C//2)
            hn = torch.cat([hn[:,0],hn[:,1]], dim=2) # => numLayers × batchSize × hiddenSize*2
            cn = hn.view(model.rnnLayerNum, 2, B, C//2)
            cn = torch.cat([cn[:,0],cn[:,1]], dim=2) # => numLayers × batchSize × hiddenSize*2
            hn = (hn,cn)
        else:
            hn = hn.view(model.rnnLayerNum, 2, B, C//2)
            hn = torch.cat([hn[:,0],hn[:,1]], dim=2) # => numLayers × batchSize × hiddenSize*2x = torch.cat([x, eos], dim=1)
        y, _ = model.dSeqRNN(y[:,1:], h0=hn) # => batchSize × seqLen × hiddenSize*2
        x = torch.cat([x,y], dim=1) # => batchSize × (seqLen+1+maxItems+1) × feaSize
        if torch.cuda.device_count() > 1:
            x,_ = nn.parallel.data_parallel(model.transformer,x)
        else:
            x,_ = model.transformer(x) # => batchSize × (seqLen+1+maxItems+1) × feaSize
        y = x[:,-model.maxItems-2:-1]
        y = model.fcLinear(F.relu(y)) # => 1 × (maxItems+1) × classNum
        
        yp1,yp2 = y[:,:,:model.classNum1],y[:,:,model.classNum1:]
        
        
        y = torch.cat([torch.tensor([[[2,2]]]*bs, device=model.device),torch.cat([yp1.argmax(dim=-1,keepdim=True),yp2.argmax(dim=-1,keepdim=True)],dim=-1)], dim=1) # => 1 × (maxItems+1)

        imgLab = torch.tensor(y, dtype=torch.long, device=model.device)

        if cnt>model.maxItems:
            break
        cnt += 1    
    Y_pre = imgLab
    Y_pre = [[(model.id2labType[j[0]]+'-'+model.id2labLoc[j[1]]).replace('<EOS>-<EOS>', '<EOS>').replace('<PAD>-<PAD>', '<PAD>').replace('<SOS>-<SOS>', '<SOS>') for j in i] for i in Y_pre]
    Y_pre = np.array([[model.lab12id[j] if j in model.lab12id else -1 for j in i] for i in Y_pre], dtype='int32')
    
    return Y_pre.tolist()


#预测值
bs = 64
ypreList = []
for i in tqdm(range((len(dataClass.testIdList)+bs-1)//(bs))):
    samples = dataClass.testIdList[i*bs:(i+1)*bs]
    imgArr = dataClass.imgArr[samples]-0.5
    yps = batch_image_predict(imgArr)
    for yp in yps:
        ypreList.append(yp[:yp.index(1)] if 1 in yp else yp)


#实际值
yList = []
for i in dataClass.testIdList:
    y = dataClass.labArr[i].tolist()
    yList.append(y[:y.index(1)])

#计算编辑距离
def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def edit_distance(a,b,EOS=1,PAD=1):
    _a = [s for s in a if s != EOS and s != PAD]
    _b = [s for s in b if s != EOS and s != PAD]

    return levenshtein(_a,_b)


def seqErrorDivSeq(Y_pre, Y):
    seqError = 0
    for yp, yt in zip(Y_pre, Y):
        l = len(yt)
        i1,i2 = yp.index(1) if 1 in yp else l,yt.index(1) if 1 in yt else l
        yp,yt = yp[:i1],yt[:i2]
        if len(yp)==l:
            if yp==yt:
                seqError += 1
    print(seqError, len(Y))
    return 1-seqError / len(Y)
def editDistDivSeq(Y_pre, Y):
    editDist = 0
    for yp, yt in zip(Y_pre, Y):
        l = len(yt)
        i1,i2 = min(yp.index(1) if 1 in yp else l, yp.index(0) if 0 in yp else l),min(yt.index(1) if 1 in yt else l,yt.index(0) if 0 in yt else l)
        yp,yt = yp[:i1],yt[:i2]
        editDist += edit_distance(yp, yt)
    print(editDist, len(Y))
    return editDist / len(Y)
def editDistDivSymbol(Y_pre, Y):
    editDist = 0
    allSymbol = 0
    for yp, yt in zip(Y_pre, Y):
        l = len(yt)
        i1,i2 = min(yp.index(1) if 1 in yp else l, yp.index(0) if 0 in yp else l),min(yt.index(1) if 1 in yt else l,yt.index(0) if 0 in yt else l)
        yp,yt = yp[:i1],yt[:i2]
        editDist += edit_distance(yp, yt)
        allSymbol += len(yt)
    print(editDist, allSymbol)
    return editDist / allSymbol


print( seqErrorDivSeq(ypreList, yList) )
print( editDistDivSeq(ypreList, yList) )
print( editDistDivSymbol(ypreList, yList) )