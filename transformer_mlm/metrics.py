import numpy as np
from sklearn import metrics as skmetrics
import warnings,os
warnings.filterwarnings("ignore")
    
class Metrictor:
    def __init__(self, metrics_file="tmp0.npz"):
        self._reporter_ = {"ACC":self.ACC, "AUC":self.AUC, "LOSS":self.LOSS, "MSE":self.MSE, "MAE":self.MAE, "seqErrorDivSeq":self.seqErrorDivSeq, "editDistDivSeq":self.editDistDivSeq, "editDistDivSymbol":self.editDistDivSymbol}
        self.metrics_file=metrics_file
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res
    
    def set_data(self, Y_pre, Y, Y_prob_pre=None, threshold=0.5, editDist=True):
        Y1 = Y
        Y1_pre = Y_pre
        self.Y1 = Y1
        self.Y1_pre = Y1_pre
#         print(self.Y1_pre)
#         print(self.Y1)
        if editDist:
            self.editDist = self.fastEditDist()
        else:
            self.editDist = None

        #if Y.shape[1] == Y_prob_pre.shape[1]+1:
        #    Y = Y[:,:-1]
        #Y = Y.reshape(-1)
        #Y_prob_pre = Y_prob_pre.reshape(len(Y),-1)
        #Y_pre = Y_prob_pre.argmax(axis=-1).reshape(-1)
        
#         self.Y,self.Y_prob_pre,self.Y_pre = None, None, None#Y[Y>0],Y_prob_pre[Y>0],Y_pre[Y>0]
        
    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')

    def ACC(self):
        return ACC(self.Y_pre, self.Y)
    def AUC(self):
        return AUC(self.Y_prob_pre,self.Y)
    def LOSS(self):
        return LOSS(self.Y_prob_pre,self.Y)
    def MSE(self):
        return MSE(self.Y_pre, self.Y)
    def MAE(self):
        return MAE(self.Y_pre, self.Y)
    
    
    def seqErrorDivSeq(self):
        return seqErrorDivSeq(self.Y1_pre, self.Y1)
    def fastEditDist(self):
        np.savez(self.metrics_file, Y_pre=self.Y1_pre, Y=self.Y1)
        os.system(f'python multi_thread_edit_dist.py --n_jobs 8 --file_name %s' % (self.metrics_file))
        editDist = int(np.load(self.metrics_file)['editDist'])
        return editDist
    def editDistDivSeq(self):
        return editDistDivSeq(self.Y1_pre, self.Y1, self.editDist)
    def editDistDivSymbol(self):
        return editDistDivSymbol(self.Y1_pre, self.Y1, self.editDist)
    
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

def seqErrorDivSeq(Y_pre, Y):
    seqError = 0
    l = Y.shape[1]
    for yp, yt in zip(Y_pre, Y):
        yp,yt = list(yp),list(yt)
        i1,i2 = yp.index(1) if 1 in yp else l,yt.index(1) if 1 in yt else l
        yp,yt = yp[:i1],yt[:i2]
        if len(yp)==len(yt):
            if yp==yt:
                seqError += 1
    return 1-seqError / len(Y)

def editDistDivSeq(Y_pre, Y, editDist=None):
    if editDist is None:
        editDist = 0
        l = Y.shape[1]
        for yp, yt in zip(Y_pre, Y):
            yp,yt = list(yp),list(yt)
            i1,i2 = min(yp.index(1) if 1 in yp else l, yp.index(0) if 0 in yp else l),min(yt.index(1) if 1 in yt else l,yt.index(0) if 0 in yt else l)
            yp,yt = yp[:i1],yt[:i2]
            editDist += edit_distance(yp, yt)
    else:
        editDist = editDist
    return editDist / len(Y)
def editDistDivSymbol(Y_pre, Y, editDist=None):
    if editDist is None:
        editDist = 0
    else:
        editDist = editDist
    allSymbol = 0
    l = Y.shape[1]
    for yp, yt in zip(Y_pre, Y):
        yp,yt = list(yp),list(yt)
        i1,i2 = min(yp.index(1) if 1 in yp else l, yp.index(0) if 0 in yp else l),min(yt.index(1) if 1 in yt else l,yt.index(0) if 0 in yt else l)
        yp,yt = yp[:i1],yt[:i2]
        if editDist is None:
            editDist += edit_distance(yp, yt)
        allSymbol += len(yt)
    return editDist / allSymbol


def ACC(Y_pre, Y):
    return (Y_pre==Y).sum() / len(Y)

def AUC(Y_prob_pre, Y):
    return skmetrics.roc_auc_score(Y, Y_prob_pre)

def LOSS(Y_prob_pre, Y):
    Y_prob_pre[Y_prob_pre>0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre<0.01] += 1e-3
    #return -np.mean(Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre))
    return -np.mean(np.log(Y_prob_pre[range(len(Y)),Y]))
    
def MSE(Y_pre, Y):
    #print(Y_pre)
    #print(Y)
    #print('\n')
    return np.mean((Y_pre-Y)**2)

def MAE(Y_pre, Y):
    return np.mean(np.abs(Y_pre-Y))
