from numpy import ndarray
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import math
from scipy import optimize
from scipy.optimize import curve_fit

with open('wordlist.txt') as wl, open('corpus_a.txt') as c:
    #Split the words in txt
    wl=[word.strip('\n') for word in wl.readlines()]
    c=c.readlines()
    c=[word.split( ) for word in c][0]
    c.pop( )

#Match the words with their frequency
dictC=Counter(c)
y=[(int(dictC[word])) for word in wl]
y=np.array(y)
x=[i for i in range(1,len(wl)+1)]
x=np.array(x)

def my_model(x: ndarray) -> ndarray:  # DO NOT modify the name
    a=[-0.31363134282167915, 1.0020934105783403, -3053.832765905848, 257397.89311928707, -375549.6960943136, 233283.9124993596]
    y = a[0]*x + np.log(x)/np.log(a[1])
    for i in range(2,len(a)):
        y += a[i]/x ** (i-2)
    return y

y_pred=my_model(x)
sse=sum([(y[i]-y_pred[i])**2 for i in range(len(y))])
print('SSE=',sse)
print('MSE=',sse/len(x))
plt.figure
plt.ylim(0,y[0])
plt.plot(x,y,'b',label = 'Real value')
plt.plot(x,y_pred,'--r',label = 'Prediction')

plt.show()
