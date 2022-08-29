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
    def calculate_y_logit(self, X, XLen):
        pass
        
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
        

    def train(self, dataClass, trainSize=256, batchSize=256, validSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, 
              optimType='Adam', preheat=0, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, isHigherBetter=False, metrics="editDistDivSeq", report=["seqErrorDivSeq", "seqErrorDivSeq", "editDistDivSymbol"], 
              savePath='model'):
        dataClass.describe()
        assert batchSize%trainSize==0  #assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        metrictor = Metrictor()
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        self.preheat()
        optimizer = self.get_optimizer(optimType=optimType, lr=lr1, weightDecay=weightDecay,momentum=momentum)
        
        schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=20, verbose=True)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0 if isHigherBetter else 9999999999,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()
        print('Start pre-heat training:')
        for e in range(epoch):
            if e==preheat:
                if preheat>0:
                    self.load(savePath+'.pkl') #得到最高的 ACC 和对应的 epoch
                self.normal()  #保留梯度信息
                optimizer = self.get_optimizer(optimType=optimType, lr=lr2, weightDecay=weightDecay,momentum=momentum)
                #self.schedulerWU = ScheduledOptim(optimizer, lr2, 1000)
                schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=30, verbose=True)
                print('Start normal training: ')
            pbar = tqdm(range(itersPerEpoch))
            for i in pbar:
                self.to_train_mode()
                X,Y = next(trainStream)
                loss = self._train_step(X,Y, optimizer)
                pbar.set_description(f"Epoch {e}: training loss - {loss.data:.3f}")
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
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
                schedulerRLR.step(mtc)
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        print(f'[Total Train]',end='')
        Y_pre,Y = self.calculate_y_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        metrictor(report)
        print(f'[Total Valid]',end='')
        Y_pre,Y = self.calculate_y_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        res = metrictor(report)
        #metrictor.each_class_indictor_show(dataClass.id2lab)
        print(f'================================')
        return res

    def preheat(self):
        for param in self.finetunedEmbList.parameters():
            param.requires_grad = False #当前参数是否需要在计算中保留对应的梯度信息


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

        
    def normal(self):
        for param in self.finetunedEmbList.parameters():
            param.requires_grad = True


    def get_optimizer(self, optimType, lr, weightDecay, momentum):
        if optimType=='Adam':
            return torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType=='AdamW':
            return torch.optim.AdamW(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType=='SGD':
            return torch.optim.SGD(self.moduleList.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)
        elif optimType=='Adadelta':
            return torch.optim.Adadelta(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType=='Lamb':
            return lamb.Lamb(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)


    def to_train_mode(self):
        for module in self.moduleList:
            module.train()  #set the module in training mode
            

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


    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = out['y_logit'].reshape(len(Y),-1)
        
        return self.criterion(Y_logit, Y)

    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy().astype('float32'),Y.cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    
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

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)


###--------------14、calculate_y_prob--------------------###
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)['y_logit']
        return F.softmax(Y_pre, dim=-1)
    # def calculate_y(self, X):
    #     Y_pre = self.calculate_y_prob(X)
    #     return torch.argmax(Y_pre, dim=1)
        
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)

    
class Trans2trans_multi_output(BaseClassifier):
    def __init__(self, classNum1, classNum2, seqMaxLen, labType2id, id2labType, labLoc2id, id2labLoc, lab12id, id2lab1, 
                 imgHeight=64, contextSizeList=[1,7,49], 
                 feaSize=192, maxItems=20, rnnLayerNum=1,
                 transNum=1, dk=64, multiNum=8, usePos=True, usePreLN=True, 
                 embDropout=0.2, tknDropout=0.5, hdnDropout=0.2, fcDropout=0.5, device=torch.device('cuda')):
        self.labType2id, self.id2labType, self.labLoc2id, self.id2labLoc, self.lab12id, self.id2lab1 = labType2id, id2labType, labLoc2id, id2labLoc, lab12id, id2lab1
        self.maxItems = maxItems
        self.seqCNN = TextCNN(imgHeight, feaSize//len(contextSizeList), contextSizeList, reduction='none', actFunc=nn.ReLU, bn=False).to(device)
        self.eSeqRNN = TextLSTM(feaSize, feaSize//2, num_layers=rnnLayerNum, dropout=hdnDropout, bidirectional=True, name='encoder').to(device)
        self.dSeqRNN = TextLSTM(feaSize, feaSize, num_layers=rnnLayerNum, dropout=hdnDropout, bidirectional=False, name='decoder').to(device)
        self.symEmbedding1 = TextEmbedding(np.random.normal(size=(classNum1, feaSize)), tknDropout=tknDropout, embDropout=embDropout, freeze=False, name='symEmbedding1').to(device)
        self.symEmbedding2 = TextEmbedding(np.random.normal(size=(classNum2, feaSize)), tknDropout=tknDropout, embDropout=embDropout, freeze=False, name='sysEmbedding2').to(device)
        if usePreLN:
            self.transformer = TextPreLNTransformer_seq2seq(seqMaxLen, layersNum=transNum, featureSize=feaSize, dk=dk, multiNum=multiNum, maxItems=maxItems, dropout=hdnDropout, usePos=usePos, name='textTransformer').to(device)
        else:    
            self.transformer = TextTransformer_seq2seq(seqMaxLen, layersNum=transNum, featureSize=feaSize, dk=dk, multiNum=multiNum, maxItems=maxItems, dropout=hdnDropout, usePos=usePos, name='textTransformer').to(device)
        self.fcLinear = MLP(feaSize, classNum1+classNum2, dropout=fcDropout, inDp=True).to(device)
        self.moduleList = nn.ModuleList([self.seqCNN, self.eSeqRNN, self.dSeqRNN, self.symEmbedding1, self.symEmbedding2, self.transformer, self.fcLinear])
        self.finetunedEmbList = nn.ModuleList([])
        self.hdnDropout= hdnDropout
        self.rnnLayerNum = rnnLayerNum
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.classNum1,self.classNum2 = classNum1,classNum2
        
    def calculate_y_logit(self, X):
        x = X['imgArr'].transpose(1,2) # => transpose(1,2)后，batchSize × seqLen × feaSize
        x_cnned = torch.cat(self.seqCNN(x), dim=-1) # => batchSize × seqLen × feaSize
        x_cnned = F.adaptive_max_pool1d(x_cnned.transpose(1,2), self.maxItems*4).transpose(1,2)
        #x_cnned = F.dropout(x_cnned, self.hdnDropout)
        y = X['imgLab_multi_output'] # => batchSize × seqLen
        y1 = self.symEmbedding1(y[:,:,0])
        y2 = self.symEmbedding2(y[:,:,1])
        y = y1+y2
        # x: batchSize × seqLen × feaSize
        #print(x.shape, y.shape)
        B,L,C = x_cnned.shape
        x_rnned,hn = self.eSeqRNN(torch.cat([x_cnned,y[:,:1]], dim=1)) # => batchSize × seqLen × hiddenSize*2, numLayers*2 × batchSize × hiddenSize
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
        #print(x.shape, y.shape)
        x = torch.cat([x, y], dim=1) # => batchSize × (seqLen+1+maxItems+1) × feaSize
        #x = F.dropout(x, self.hdnDropout)
        if torch.cuda.device_count() > 1:
            x,_ = nn.parallel.data_parallel(self.transformer,x)
        else:
            x,_ = self.transformer(x) # => batchSize × (seqLen+1+maxItems+1) × feaSize
        y = x[:,-self.maxItems-2:-1]
        y = F.relu(y)
        return {'y_logit':self.fcLinear(y)} # => batchSize × (maxItems+1) × classNum
    
    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        Y1 = Y["imgLab_multi_output"][:,:,0].reshape(-1)
        Y2 = Y["imgLab_multi_output"][:,:,1].reshape(-1)
        Y_logit1 = out['y_logit'][:,:,:self.classNum1].reshape(len(Y1),-1)
        Y_logit2 = out['y_logit'][:,:,self.classNum1:].reshape(len(Y2),-1)
        
        return self.criterion(Y_logit1, Y1) + self.criterion(Y_logit2, Y2)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy().astype('float32'),Y["imgLab_multi_output"].cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy().astype('float32'),Y["imgLab"].cpu().data.numpy().astype('int32')
            YArr.append(Y)
            
            Y_pre1,Y_pre2 = Y_pre[:,:,:self.classNum1].argmax(axis=-1),Y_pre[:,:,self.classNum1:].argmax(axis=-1)
            Y_pre = np.concatenate([Y_pre1[:,:,np.newaxis],Y_pre2[:,:,np.newaxis]], axis=-1).astype('int32')
            Y_pre = [[(self.id2labType[j[0]]+'-'+self.id2labLoc[j[1]]).replace('<EOS>-<EOS>', '<EOS>').replace('<PAD>-<PAD>', '<PAD>').replace('<SOS>-<SOS>', '<SOS>') for j in i] for i in Y_pre]
            Y_pre = np.array([[self.lab12id[j] if j in self.lab12id else -1 for j in i] for i in Y_pre], dtype='int32')
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)['y_logit']
        Y_pre1,Y_pre2 = Y_pre[:,:,:self.classNum1].argmax(axis=-1),Y_pre[:,:,self.classNum1:].argmax(axis=-1)
        Y_pre1,Y_pre2 = F.softmax(Y_pre1, dim=-1),F.softmax(Y_pre2, dim=-1)
        return torch.cat([Y_pre1,Y_pre2], dim=-1)
    

