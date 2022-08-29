import numpy as np
import sys
import os,torch,math
from utils import *
from DL_ClassifierModel import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #选择需要使用的GPU
torch.cuda.device_count()

parser = argparse.ArgumentParser(description='Predict.')
parser.add_argument('-encoding', dest='encoding', type=str, required=True, help='Encoding type (semantic or agnostic).')
args = parser.parse_args()

# 读取实际数据
dataClass = DataClass_Batch('../DataSet/', 'Data/train.txt', 'Data/test.txt', 'Data/vocabulary_agnostic.txt', 'Data/vocabulary_semantic.txt', encoding=args.encoding, hs=128, dropBLANK=True)

#Trans2Trans模型
model = Trans2trans(classNum=len(dataClass.label), seqMaxLen=dataClass.maxItems*4+1+dataClass.maxItems+1,
                    tknDropout=0.1, embDropout=0.0, hdnDropout=0.1, fcDropout=0.0,
                    imgHeight=128, contextSizeList=[1,5,25,49], feaSize=1024, rnnLayerNum=2,
                    transNum=4, dk=48, multiNum=8, usePos=True, usePreLN=True,
                    maxItems=dataClass.maxItems, device=torch.device('cuda'))


#加载已训练好的模型
model.load('TrainModel_xxx.pkl', map_location="cuda")
model.to_eval_mode()

# 每次取一批数据
def batch_image_predict(batchImgArr):
    bs = len(batchImgArr)
    imgArr = torch.tensor(batchImgArr-0.5, dtype=torch.float32, device=model.device)
    imgLab = torch.cat([torch.tensor([[2]]*bs,device=model.device),torch.zeros((bs,model.maxItems+1), dtype=torch.long, device=model.device)], dim=1)
    
    cnt = 1
    imgArr = imgArr.transpose(1,2)
    x = torch.cat(model.seqCNN(imgArr), dim=-1)
    x_pool = F.adaptive_max_pool1d(x.transpose(1,2), model.maxItems*4).transpose(1,2)
    while True:
        
        y = model.symEmbedding(imgLab)
        B,L,C = x_pool.shape  # x_pool: 1 × seqLen × feaSize
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
        
        y = torch.cat([torch.tensor([[2]]*bs, device=model.device),y.argmax(dim=-1)], dim=1) # => 1 × (maxItems+1)
        imgLab = torch.tensor(y, dtype=torch.long, device=model.device)

        if cnt>model.maxItems or y[0,cnt]==1:
            break
        cnt += 1 
    yp = imgLab.tolist()
    return yp

# 在实际测试集上取一批图像得到预测标签结果
testBatch = 64
testIdList = dataClass.testIdList
ypreList = []

for i in range((len(testIdList)+testBatch-1)//testBatch):
 
    imgArr = dataClass.imgArr[testIdList[i*testBatch:(i+1)*testBatch]]
    maxW = -1
    for img in imgArr:
        if len(img[0]) > maxW:
            maxW = len(img[0])
    imgs = []
    for img in imgArr:
        imgs.append( np.hstack([np.vstack(img),np.zeros((len(img),maxW-len(img[0])), dtype='bool')]).astype('bool') )
    imgs = np.array(imgs, dtype='bool')
    yp = batch_image_predict(imgs)
    for y in yp:
        ypreList.append( y[:y.index(1)] if 1 in y else y )


# 真实测试集标签
yList = []
for i in dataClass.testIdList:
    y = dataClass.labArr[i].tolist()
    yList.append(y[:y.index(1)])


# 计算编辑距离
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

def edit_distance(a,b):
    return levenshtein(a,b)

seqAcc = 0
seqNum = 0
editDistance = 0
symbolNum = 0
for yp,yt in zip(ypreList, yList):
    yp_l,yt_l = len(ypreList),len(yList)
    yp_loc,yt_loc = yp.index(1) if 1 in yp else yp_l, yt.index(1) if 1 in yt else yt_l
    yp,yt = yp[:yp_loc], yt[:yt_loc]
    if len(yp) == len(yt):
        if yp == yt:
            seqAcc += 1
    edit_dist = edit_distance(yp, yt)
    editDistance += edit_dist
    symbolNum += (len(yt)-1)
    seqNum += 1

print(seqNum, editDistance, symbolNum)
print('seqError/seqNum = %.5lf    '%(1-seqAcc/seqNum) )
print('editDistance/seqNum = %.5lf    '%(editDistance/seqNum) )
print('editDistance/symbolNum = %.5lf    '%(editDistance/symbolNum) )