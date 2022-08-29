import multiprocessing,argparse
import numpy as np

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


def thread(data):
    yp,yt,editDist = data
    l = len(yt)
    yp,yt = list(yp),list(yt)
    i1,i2 = min(yp.index(1) if 1 in yp else l, yp.index(0) if 0 in yp else l),min(yt.index(1) if 1 in yt else l,yt.index(0) if 0 in yt else l)
    yp,yt = yp[:i1],yt[:i2]
    dist = levenshtein(yp, yt)
    lock.acquire()
    editDist.value += dist
    lock.release()

def init_lock(l):
    global lock
    lock = l

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', default='8')
parser.add_argument('--file_name', default='tmp0.npz')
args = parser.parse_args()
if __name__ == '__main__':
    data = np.load(args.file_name)
    Y_pre,Y = data['Y_pre'],data['Y']
    editDist = multiprocessing.Manager().Value('i', 0)
    l = multiprocessing.Lock()
    pool=multiprocessing.Pool(int(args.n_jobs), initializer=init_lock, initargs=(l, ))
    pool.map(thread, zip(Y_pre, Y, [editDist]*len(Y)))
    pool.close()
    pool.join()
    np.savez(args.file_name, editDist=editDist.value)