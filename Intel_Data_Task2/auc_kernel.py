import numpy as np

def updateBuffer(buff, x, n_buff, n_buff_t):
    if(len(buff) < n_buff):
        buff.append(x)
    else:
        z = np.random.random()
        if(z < float(n_buff / n_buff_t)):
            idx = np.random.randint(0, len(buff))
            buff[idx] = x
    return buff

def updateClassifier(w, x, y, C, buff):
    for sample in buff:
        if(y * np.dot(w.T, (x - sample)) <= 1):
            grad = C * y * (x - sample) / 2
            grad = grad[:, np.newaxis]
            w = w + grad
    return w

def onlineAucMaximum(x, y, n_pos, n_neg, C = 1.0):
    w = np.zeros((x.shape[1], 1))
    n_pos_t = 0
    n_neg_t = 0
    b_pos = []
    b_neg = []
    for idx in range(len(y)):
        if(y[idx] == 1):
            n_pos_t += 1
            C_t = C * max(1, n_neg_t / n_neg)
            b_pos = updateBuffer(b_pos, x[idx], n_pos, n_pos_t)
            w = updateClassifier(w, x[idx], y[idx], C_t, b_neg)
        else:
            n_neg_t += 1
            C_t = C * max(1, n_pos_t / n_pos)
            b_neg = updateBuffer(b_neg, x[idx], n_neg, n_neg_t)
            w = updateClassifier(w, x[idx], y[idx], C_t, b_pos)
    return w