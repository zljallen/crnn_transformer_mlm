import numpy as np
import pandas as pd
import torch,time,os,pickle,random
from torch import nn as nn
from nnLayer import *
from metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
from tqdm import tqdm
from torchvision import models
from pytorch_lamb import lamb


class BaseClassifier:
    def __init__(self):
        pass
        

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
        

#     def preheat(self):
#         for param in self.finetunedEmbList.parameters(): #遍历整个模型中的参数
#             param.requires_grad = False #当前参数是否需要在计算中保留对应的梯度信息
# ###--------------------------------------------4、normal------------------------------------------###
#     def normal(self):
#         for param in self.finetunedEmbList.parameters():
#             param.requires_grad = True #自动求导开始并记录对Tensor的操作
####-----------------------------------------5、get_optimizer：优化器--------------------------------######
    def get_optimizer(self, optimType, lr, weightDecay, momentum):
        if optimType=='Adam':
            return torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType=='AdamW':
            return torch.optim.AdamW(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType=='Lamb':
            return lamb.Lamb(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType=='SGD':
            return torch.optim.SGD(self.moduleList.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)
        elif optimType=='Adadelta':
            return torch.optim.Adadelta(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)            
####-----------------------------------------6、to_train_mode-----------------------------------####
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()  #set the module in training mode

###---------------------------------------7、calculate_loss-------------------------------------###
    def calculate_loss(self, X, Y):
        # print(X['imgArr'].shape, X['imgLab'].shape, X['inputLen'], X['targetLen'])
        # print(Y.shape)
        out = self.calculate_y_logit(X) #得到预测输出out ( batchSize*(maxItems+1)*len(dataClass.id2lab2) )
        Y = Y.reshape(-1) #得到实际标签Y
        Y_logit = out['y_logit'].reshape(len(Y),-1)
        return self.criterion(Y_logit, Y) # 计算真实值与预测值之间的交叉熵损失
####-----------------------------------------8、_train_step------------------------------####
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        # print(loss)
        loss.backward()
        
        if p:
            nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            
        return loss*self.stepUpdate        
#####--------------------------------------9、to_eval_mode------------------------------------######
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
#####-------------------------------------10、calculate_y_prob---------------------------------------------#####
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)['y_logit']
        return F.softmax(Y_pre, dim=-1)
    # def calculate_y(self, X):
    #     Y_pre = self.calculate_y_prob(X)
    #     return torch.argmax(Y_pre, dim=1)
#####--------------------------------------11、calculate_y_by_iterator------------------------------------######
    def calculate_y_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy().argmax(axis=-1).astype('float32'),Y.cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr            
#####---------------------------------------12、calculate_y_prob_by_iterator---------------------------######
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy().argmax(axis=-1).astype('float32'),Y.cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
    
    
        
#####-------------------------------------14、train----------------------------------######
    def train(self, dataClass, trainSize=256, batchSize=256, validSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, optimType='Adam', warmupEpochs=1, lr=0.00003, momentum=0.9, weightDecay=0, isHigherBetter=False, metrics="editDistDivSeq", report=["seqErrorDivSeq", "editDistDivSeq", "editDistDivSymbol"], savePath='model', metrics_savefile="tmp0.npz"):
        
        dataClass.describe()
        assert batchSize%trainSize==0 #assert(断言)用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        metrictor = Metrictor(metrics_file=metrics_savefile)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize

        # mini-batch读取训练集数据
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)

        # 训练一次完整的训练集，需要进行的次数
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
       
        #设置参数优化器Adam
        optimizer = self.get_optimizer(optimType=optimType, lr=lr, weightDecay=weightDecay, momentum=momentum)
        
        # 在达到一定条件下降低学习率
#         schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=20, verbose=True)
        warmSteps = int(itersPerEpoch * warmupEpochs)
        decaySteps = int(itersPerEpoch*epoch) - warmSteps
        schedulerRLR = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i:i/warmSteps if i<warmSteps else (decaySteps-(i-warmSteps))/decaySteps)
        
        # 赋值操作 mtc=0.0, bestMtc=9999999999, stopSteps=0
        mtc,bestMtc,stopSteps = 0.0,0.0 if isHigherBetter else 9999999999,0
        
        # mini-batch读取验证集数据
        if dataClass.validSampleNum>0: 
            validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        
        #记录训练开始时间
        st = time.time()
        
        for e in range(epoch):
            print(f"Epoch {e+1} with learning rate {optimizer.state_dict()['param_groups'][0]['lr']:.6f}...")

            for i in tqdm(range(itersPerEpoch)):
                #设置模型为训练模式
                self.to_train_mode()
                X,Y = next(trainStream)
                loss = self._train_step(X,Y, optimizer) #得到交叉熵损失
                schedulerRLR.step()
#                 pbar.set_description(f"Training Loss: {loss}; Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")

                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    with torch.no_grad():
                        self.to_eval_mode()
                    
                        if dataClass.validSampleNum>0:
                            X,Y = next(validStream)
                            loss = self.calculate_loss(X,Y)
                            print(f' [valid] loss= {loss:.3f};', end='')
#                     restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
#                     speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
#                     print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                with torch.no_grad():
                    # 设置为非训练模式
                    self.to_eval_mode()
                    print(f'========== Epoch:{e+1:5d} ==========')
                    print(f'[Total Train]',end='')
                    Y_pre,Y = self.calculate_y_by_iterator(dataClass.one_epoch_batch_data_stream(validSize, type='train', device=self.device))                   
                    metrictor.set_data(Y_pre, Y)
                    metrictor(report)
                    
                    print(f'[Total Valid]',end='')
                    Y_pre,Y = self.calculate_y_by_iterator(dataClass.one_epoch_batch_data_stream(validSize, type='valid', device=self.device))                
                    metrictor.set_data(Y_pre, Y)
                    res = metrictor(report)
                    mtc = res[metrics]

                    print('=================================')
                    if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                        print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                        bestMtc = mtc
                        bestEpoch = e+1
                        self.save("%s.pkl"%(savePath), bestEpoch, bestMtc, dataClass)
                        stopSteps = 0
                    else:
                        stopSteps += 1
                        if stopSteps>=earlyStop:
                            print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                            break
        self.load("%s.pkl"%(savePath))
        os.rename("%s.pkl"%(savePath), "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)))
        
        with torch.no_grad():
            print(f'============ Result ============')
            print(f'[Total Train]',end='')
            Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
            metrictor.set_data(Y_pre, Y)
            metrictor(report)
            print(f'[Total Valid]',end='')
            Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
            metrictor.set_data(Y_pre, Y)
            res = metrictor(report)
            #metrictor.each_class_indictor_show(dataClass.id2lab)
            print(f'================================')
        return res
    

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        # print(self.moduleList)
        print([i] for i in self.moduleList)
        
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)


    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            if "trainIdList" in parameters:
                dataClass.trainIdList = parameters['trainIdList']
            if "validIdList" in parameters:
                dataClass.validIdList = parameters['validIdList']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

# Trans2Trans模型
class Trans2trans(BaseClassifier):
    def __init__(self, classNum, seqMaxLen, imgHeight=128, contextSizeList=[1,5,25,49], 
                 feaSize=192, maxItems=20, rnnLayerNum=1,
                 transNum=1, dk=64, multiNum=8, usePos=True, usePreLN=True, 
                 embDropout=0.2, tknDropout=0.5, hdnDropout=0.2, fcDropout=0.5, device=torch.device('cuda')):
        
        self.maxItems = maxItems 
        self.seqCNN = TextCNN(imgHeight, feaSize//len(contextSizeList), contextSizeList, reduction='none', actFunc=nn.ReLU, bn=True).to(device)
    
        self.eSeqRNN = TextLSTM(feaSize, feaSize//2, num_layers=rnnLayerNum, dropout=hdnDropout, bidirectional=True, name='encoder').to(device)
        self.dSeqRNN = TextLSTM(feaSize, feaSize, num_layers=rnnLayerNum, dropout=hdnDropout, bidirectional=False, name='decoder').to(device)             
        self.symEmbedding = TextEmbedding(np.random.normal(size=(classNum, feaSize)), tknDropout=tknDropout, embDropout=embDropout, freeze=False).to(device)

        if usePreLN:
            self.transformer = TextPreLNTransformer_seq2seq(seqMaxLen, layersNum=transNum, featureSize=feaSize, dk=dk, multiNum=multiNum, maxItems=maxItems, dropout=hdnDropout, usePos=usePos, name='textTransformer').to(device)
    
        else:
            self.transformer = TextTransformer_seq2seq(seqMaxLen, layersNum=transNum, featureSize=feaSize, dk=dk, multiNum=multiNum, maxItems=maxItems, dropout=hdnDropout, usePos=usePos, name='textTransformer').to(device)
        self.fcLinear = MLP(feaSize, classNum, dropout=fcDropout, inDp=True).to(device)
        self.moduleList = nn.ModuleList([self.seqCNN, self.eSeqRNN, self.dSeqRNN, self.symEmbedding, self.transformer, self.fcLinear])
        self.finetunedEmbList = nn.ModuleList([])
        self.hdnDropout= hdnDropout
        self.rnnLayerNum = rnnLayerNum
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def calculate_y_logit(self, X):

        x = X['imgArr'].transpose(1,2) # => transpose(1,2)后，batchSize × seqLen × feaSize  
        x_cnned = torch.cat(self.seqCNN(x), dim=-1) # => batchSize × seqLen × feaSize  
        x_cnned = F.adaptive_max_pool1d(x_cnned.transpose(1,2), self.maxItems*4).transpose(1,2)
        x_cnned = F.dropout(x_cnned, self.hdnDropout)
        
        y = X['imgLab'] # => batchSize × seqLen
        y = self.symEmbedding(y)  # => batchSize × seqLen × feaSize

        # x: batchSize × seqLen × feaSize
        B,L,C = x_cnned.shape
        x_rnned,hn = self.eSeqRNN(torch.cat([x_cnned,y[:,:1]], dim=1)) # x_rnned=> batchSize × seqLen × hiddenSize*2, hn=> numLayers*2 × batchSize × hiddenSize

        eos = x_rnned[:,-1:]
        x_rnned = x_rnned[:,:-1]
        x = x_rnned + x_cnned
        x = torch.cat([x, eos], dim=1)
        if type(hn)==tuple:
            hn,cn = hn
            hn = hn.view(self.rnnLayerNum, 2, B, C//2)
            hn = torch.cat([hn[:,0],hn[:,1]], dim=2) # => numLayers × batchSize × hiddenSize*2
            cn = hn.view(self.rnnLayerNum, 2, B, C//2)
            cn = torch.cat([cn[:,0],cn[:,1]], dim=2) # => numLayers × batchSize × hiddenSize*2
            hn = (hn,cn)
        else:
            hn = hn.view(self.rnnLayerNum, 2, B, C//2)
            hn = torch.cat([hn[:,0],hn[:,1]], dim=2) # => numLayers × batchSize × hiddenSize*2
        y,_ = self.dSeqRNN(y[:,1:], h0=hn) # => batchSize × seqLen × hiddenSize*2

        x = torch.cat([x, y], dim=1) # => batchSize × (seqLen+1+maxItems*4+1) × feaSize

        #x = F.dropout(x, self.hdnDropout)
        if torch.cuda.device_count() > 1:
            x,_ = nn.parallel.data_parallel(self.transformer,x)
        else:
            x,_ = self.transformer(x) # => batchSize × (seqLen+1+maxItems*4+1) × feaSize

        y = x[:,-self.maxItems-2:-1]
        y = F.relu(y)

        return {'y_logit':self.fcLinear(y)} # => batchSize × (maxItems+1) × classNum
    