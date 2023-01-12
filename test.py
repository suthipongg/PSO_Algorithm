from time import time
import numpy as np
import random
from numpy import exp, sqrt, cos, sin, pi
'''
a = np.array([
    [2,1],
    [3,0]
])
b = np.sum(a, 1)
b = np.reshape(b, (2,1))
c = np.reshape(np.sum(a==b, 0), (2,1))
print(c)'''

'''
a = 0

t1 = time()*10**6
i = 0
while i < 10**9:
    a += 1
    i += 1
t1 = time()*10**6 - t1
t2 = time()*10**6
for i in range(0,10**9):
    a +=1
t2 = time()*10**6 - t2
print(t1)
print(t2)'''

'''
class test:
    def __init__(self):
        self.a = 0
        
    def tt(self):
        self.b = 5
    def call(self):
        self.b += 6

a = test()
print(a.a, a.b)
a.tt()
print(a.a, a.b)
a.call()
print(a.a, a.b)'''
'''
a = np.array([
    [2],
    [1],
    [7],
    [0],
    [8],
    [2]
])
aa = np.array([
    [2,8],
    [1,3],
    [7,6],
    [0,2],
    [8,4],
    [2,3]
])
t = a <3
aa_dom = aa*t
e = np.tile(t, (1, 6))
i = (np.identity(6) == 0)
d = (i*e).T
idx = np.random.rand(*d.shape)*d
f = np.argmax(idx, axis=1)
print(idx)
num = np.arange(6)
d[num,f] = 0
print(f)
idx = np.random.rand(*d.shape)*d
f = np.argmax(idx, axis=1)
print(idx)
print(f)
'''
'''
a = np.random.rand(3,1)
print(0.5<a)'''
a = 3
print(a)