# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:10:01 2018

@author: Astrid
"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from numpy import sqrt,pi,e,tanh,cosh
#uppgiftA
'''def formel(a,b,c):
    #a
    xp=(-b+sqrt(b**2-4*a*c))/(2*a)
    xm=(-b-sqrt(b**2-4*a*c))/(2*a)
    #b
    xbp=(2*c)/(-b+sqrt(b**2-4*a*c))
    xbm=(2*c)/(-b-sqrt(b**2-4*a*c))
    #c
    xm=(-b-sqrt(b**2-4*a*c))/(2*a)
    xbm=(2*c)/(-b-sqrt(b**2-4*a*c))
    
    
    
    
    print('Uppgift a:',xp,xm,'Uppgift b:',xbp,xbm,'Uppgift c:',xm,xbm)
    
a=0.001
b=1000
c=0.001
    
formel(a,b,c)

#uppgiftB
def f(x):
    return 0.02+1.01*x+0.1*x*x


mean,sigma1,sigma2=1,0.1,0.2
x=mean+sigma1*np.random.randn(1000000)+sigma2*np.random.randn(1000000)

df=f(x)-f(mean)
plt.figure()

sigmax=(sigma1)**2+(sigma2)**2
print(sigmax)

my=0
x2=np.linspace(-2,2,10000)
sigma=np.sqrt(sigmax*(1.01+0.2*1)**2)
print('sigma=',sigma)
y=(1/(sigma*sqrt(2*pi))*e**(-(x2-my)**2/(2*sigma**2)))
plt.plot(x2,y,color='r')

what=np.std(f(x))
print('std=',what)


n,bins,patches=plt.hist(df,100,density=1,facecolor='blue',label='Exact')

plt.legend()
plt.xlabel('f(x)-f(x_mean)')
plt.ylabel('A.u.')
plt.axis([-2, 2, 0, 2])
plt.grid(True)
plt.show()

def f1(x):
    return 0.02+1.01*x+0.8*x*x


mean,sigma1,sigma2=1,0.1,0.2
x=mean+sigma1*np.random.randn(1000000)+sigma2*np.random.randn(1000000)

df=f1(x)-f1(mean)
plt.figure()

sigmax=(sigma1)**2+(sigma2)**2
print(sigmax)

my=0
x2=np.linspace(-2,2,10000)
sigma=np.sqrt(sigmax*(1.01+1.6*1)**2)
print('sigma=',sigma)
y=(1/(sigma*sqrt(2*pi))*e**(-(x2-my)**2/(2*sigma**2)))
plt.plot(x2,y,color='r')

what=np.std(f1(x))
print('std=',what)


n,bins,patches=plt.hist(df,100,density=1,facecolor='blue',label='Exact')

plt.legend()
plt.xlabel('f(x)-f(x_mean)')
plt.ylabel('A.u.')
plt.axis([-2, 2, 0, 2])
plt.grid(True)
plt.show()'''

#uppgift 3
def matris():
    A=np.matrix([[1.01, 1, 1], [1, 1.01,1],[1,1,1.01]])
    #A_inv=np.linalg.inv(A)
    #print(A_inv)
    kappa1=np.linalg.norm(A,np.inf)*np.linalg.norm(np.linalg.inv(A),np.inf)
    print(kappa1)
    #kappa2=np.linalg.cond(A, np.inf)
    #print(kappa2)
    stör=np.matrix([0,0,0.001]).transpose()
    y=np.matrix([1,1,1]).transpose()
    kappa3=((np.linalg.norm(np.linalg.inv(A)*stör,np.inf)/np.linalg.norm(np.linalg.inv(A)*y,np.inf))/
    (np.linalg.norm(stör,np.inf)/np.linalg.norm(y,np.inf)))
    print(kappa3)
    print(np.linalg.norm(np.linalg.inv(A)*stör,np.inf))
    print('inv',np.linalg.norm(np.linalg.inv(A)*y,np.inf))
    
    
matris()

#uppgift4
#5.6
'''def f(x):
    return x**4-2*x+1

def I_1(F):
    N=10
    a=0.0
    b=2.0
    h=(b-a)/N
    
    s=0.5*f(a)+0.5*f(b)
    
    for k in range(1,N):
        s+=f(a+k*h)
    print('I_1=',h*s)
    return h*s

def I_2(F):
    N=20
    a=0.0
    b=2.0
    h=(b-a)/N
    
    s=0.5*f(a)+0.5*f(b)
    
    for k in range(1,N):
        s+=f(a+k*h)
    print('I_2=',h*s)
    return h*s

F=f(x)
I_1=I_1(F)
I_2=I_2(F)

fel=(1/3)*(I_2-I_1)
print(fel)
print(I_2-4.4)'''
#fler trapets --> bättre approximation

#5.10


'''def period(b):
    
    def T(x):
        return sqrt(8)/sqrt(b**4-x**4)
    
    #for k in range(b):
    N=20
    a=0.0    
    
    x,w=np.polynomial.legendre.leggauss(N)
    xp=0.5*(b-a)*x+0.5*(b+a)
    wp=0.5*(b-a)*w
    
    s=0.0
    for k in range(N):
        s+=wp[k]*T(xp[k])
    #s=wp[k]*T(xp[k],b[k])
        #plt.plot(x,s)
        #plt.show
    #print(s)
        
    return s
        
        #integralos(T,gaussxw)
    
    
x=np.linspace(0,2,20)
u=[]
for i in range(20):
    u.append(period(i))
plt.plot(x,u)
plt.show()

period(x)'''

#5.15
'''def f(x):
    return 1+(1/2)*tanh(2*x)

def cent_diff(x):
    h=0.1 #litet h mer nogrannhet
    derivative=((f(x+h/2)-f(x-h/2))/h)
    return derivative

def deriv(x):
    return 1/(cosh(2*x)**2)

    
x=np.linspace(-2,2,100)
#plt.figure()
plt.plot(x,cent_diff(x),'b.')
plt.plot(x,deriv(x),'r')
plt.show()'''

#uppgift 5

def e_f(x,h):
    return max((h**2/4)*abs(-2*tanh(x))/cosh(x)**2)

def skattning(x,n):
    a=0
    b=3/n
    fellista=[]
    for i in range(n):
        
        fel=tanh(x)-((b-x)*tanh(a)+(x-a)*(tanh(b)))/(b-a)
        maxima=max(fel)
        a=b
        b+=3/n
        #print(a)
        #print(b)
        
        fellista.append(maxima)
    return max(fellista)



n=10
h=3/n
x=np.linspace(0,3,100)
print(e_f(x,h),skattning(x,n))









