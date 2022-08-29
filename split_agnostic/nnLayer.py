from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict
#import torchvision as tv

      

class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, bn=False, name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
            )
        self.actFunc = actFunc()
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.batcnNorm = nn.BatchNorm1d(filterSize)
        self.bn = bn
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).transpose(1,2) for conv in self.conv1dList] # => scaleNum * (batchSize × seqLen × filterSize)

        if self.bn:
            x = [self.batcnNorm(i.transpose(1,2)).transpose(1,2) for i in x]
        x = [self.actFunc(i) for i in x]

        if self.reduction=='pool':
            x = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]  ##squeeze去掉第二维度
            return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
        elif self.reduction=='none':
            return x # => scaleNum * (batchSize × seqLen × filterSize)
        elif self.reduction=='sum':
            x = [i.unsqueeze(dim=0) for i in x]  ##unsqueeze在第0维增加一个维度增加
            return torch.cat(x,dim=0).sum(dim=0)


class TextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=True, name='textBiLSTM'):
        super(TextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, x, xlen=None, h0=None):
        # x: batchSizeh × seqLen × feaSize

        output, hn = self.biLSTM(x, h0) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize

        return output, hn # output: batchSize × seqLen × hiddenSize*2
        

class TextEmbedding(nn.Module):
    def __init__(self, embedding, tknDropout=0.3, embDropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=tknDropout)
        self.dropout2 = nn.Dropout(p=embDropout)
        self.p = tknDropout+embDropout
    def forward(self, x):
        # x: batchSize × seqLen
        if self.p>0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x

####-----------------------4.1->SelfAttention_seq2seq----------------------####
class PreLN_SelfAttention_seq2seq(nn.Module):
    def __init__(self, maxItems, featureSize, dk, multiNum, name='selfAttn'):
        super(PreLN_SelfAttention_seq2seq, self).__init__()
        self.maxItems = maxItems
        self.dk = dk
        self.multiNum = multiNum
        self.layerNorm1 = nn.LayerNorm([featureSize])
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        x = self.layerNorm1(x)
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys    = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values  = [self.WV[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        scores  = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        
        mask = torch.zeros(scores[0].shape, dtype=torch.float32, device=scores[0].device) # => batchSize × seqLen × seqLen
        mask[:,:-(self.maxItems+1), -(self.maxItems+1):] = -999999999
        for j in range(self.maxItems):
            mask[:,-(self.maxItems+1-j),-(self.maxItems+1-j):] = -999999999
            
        for i in range(len(scores)):
            scores[i] = F.softmax(scores[i]+mask, dim=2)
        self.scores = scores
        
        z = [torch.bmm(scores[i], values[i]) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batchSize × seqLen × feaSize
        return z

###------------------------------4.2->FFN---------------------------------###
class PreLN_FFN(nn.Module):
    def __init__(self, featureSize, dropout=0.1, name='FFN'):
        super(PreLN_FFN, self).__init__()
        
        self.layerNorm2 = nn.LayerNorm([featureSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize, featureSize*4), 
                        nn.ReLU(),
                        nn.Linear(featureSize*4, featureSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = x + self.dropout(z) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(self.layerNorm2(z)) # => batchSize × seqLen × feaSize
        return z+self.dropout(ffnx) # => batchSize × seqLen × feaSize
    
#####----------------------4.3->Transfofrmer_seq2seq------------------#####
class PreLN_Transformer_seq2seq(nn.Module):
    def __init__(self, maxItems, featureSize, dk, multiNum, dropout=0.1):
        super(PreLN_Transformer_seq2seq, self).__init__()
        self.selfAttn = PreLN_SelfAttention_seq2seq(maxItems, featureSize, dk, multiNum)
        self.ffn = PreLN_FFN(featureSize, dropout)

    def forward(self, input):
        x, xlen = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z = self.selfAttn(x, xlen) # => batchSize × seqLen × feaSize
        return (self.ffn(x, z),xlen) # => batchSize × seqLen × feaSize

#####----------------------4.4->TextTransfofrmer_seq2seq------------------#####
class TextPreLNTransformer_seq2seq(nn.Module):
    def __init__(self, seqMaxLen, layersNum, featureSize, dk, multiNum, maxItems=10, dropout=0.1, usePos=True, name='textTransformer'):
        super(TextPreLNTransformer_seq2seq, self).__init__()
        posEmb = [[np.sin(pos/10000**(2*i/featureSize)) if i%2==0 else np.cos(pos/10000**(2*i/featureSize)) for i in range(featureSize)] for pos in range(seqMaxLen)]
        self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, PreLN_Transformer_seq2seq(maxItems, featureSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        self.usePos = usePos
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        if self.usePos:
            x = x+self.posEmb
        x = self.dropout(x) # => batchSize × seqLen × feaSize
        return self.transformerLayers((x, xlen)) # => batchSize × seqLen × feaSize
    
    
    
#####----------------------5->MLP---------------------#####
class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, startBn=False, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, inDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        self.sBn = nn.BatchNorm1d(inSize)
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.startBn = startBn
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.inDp = inDp
    def forward(self, x):
        if self.startBn:
            x = self.sBn(x)
        if self.inDp:
            x = self.dropout(x)
        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape)==2 else bn(x.transpose(1,2)).transpose(1,2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x) if len(x.shape)==2 else self.bns[-1](x.transpose(1,2)).transpose(1,2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x
