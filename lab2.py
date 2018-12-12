
#ATT FRÅGA
#Emmelies kod på fråga 1, newtons metod
#Uppgift 2: vilken funktion
#Uppgift 3: var är matrisen?????

import numpy as np
from numpy import e, pi, log
import matplotlib.pyplot as plt
from math import exp,sqrt
import functions

G = 6.674*10**(-11) #m^3kg^(-1)s^(-2)
M = 5.974*10**(24) #kg
m = 7.348*10**(22) #kg
R = 3.844*10**8 #m
w = 2.662*10**(-6) #s^(-1)

def uppgift1():
    guess = 10
    for n in range(10):
        guess = newtons(guess)
        print(guess)

def f(r):
    f = G*M*R**2 - 2*G*M*R*r-w**2*r**3*R**2+2*w**2*R*r**4-w**2*r**5+G*M*r**2-G*m*r**2
    return f

def df(r):
    df = -2*G*M*R - 3*w**2*r**2*R**2 + 8*w**2*R*r**3-5*w**2*r**4+2*G*M*r-2*G*m*r
    return df

def newtons(x0):
    x1 = x0 -(f(x0))/(df(x0))
    return x1


def uppgift2():
    N = 100
    h = 6.6261*10**(-34)
    c = 299792458
    lambda1 = 3.9*10**(-7)
    lambda2 = 7.5*10**(-7)
    kB = 1.38064852*10**(-23)
    T = np.linspace(300,10000,N)
    a = (h*c)/(lambda2*kB*T)
    b = (h*c)/(lambda1*kB*T)
    u = []
    for i in range(N):
        u.append(functions.quadratic_gauss(N,a[i],b[i],efficiency))
    plt.plot(T,u)
    print('Maximum efficiency with numpy: ',round(T[np.argmax(u)],2), 'K')
    print('Maximum efficiency with golden ratio search: ',round(golden_ratio_search(function,300,7000),2), 'K')

def function(i):
    N = 100
    h = 6.6261*10**(-34)
    c = 299792458
    lambda1 = 3.9*10**(-7)
    lambda2 = 7.5*10**(-7)
    kB = 1.38064852*10**(-23)
    T = np.linspace(300,10000,N)
    a = (h*c)/(lambda2*kB*T)
    b = (h*c)/(lambda1*kB*T)
    u = []
    for i in range(N):
        u.append(functions.quadratic_gauss(N,a[i],b[i],efficiency))
    return -u[i]

def efficiency(x):
    eta = (15/(pi**4))*(x**3/(e**x-1))
    return eta

def golden_ratio_search(f,x1,x4):
    # Constants
    accuracy = 0.05
    z = (1+sqrt(5))/2       # Golden ratio

    # Initial positions of the four points
    x2 = x4 - (x4-x1)/z
    x3 = x1 + (x4-x1)/z
    
    # Initial values of the function at the four points
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    f4 = f(x4)

    # Main loop of the search process
    while x4-x1>accuracy:
        if f2<f3:
            x4,f4 = x3,f3
            x3,f3 = x2,f2
            x2 = x4 - (x4-x1)/z
            f2 = f(x2)
        else:
            x1,f1 = x2,f2
            x2,f2 = x3,f3
            x3 = x1 + (x4-x1)/z
            f3 = f(x3)

    return 0.5*(x1+x4)


def uppgift3():
    decay = np.array([log(50),log(33),log(31),log(14),log(20),log(13)])
    time = np.array([0.25,0.75,1.25,1.75,2.25,2.75])
    N = len(time)
    E_x = np.sum(time)/N
    E_y = np.sum(decay)/N
    E_xx = (np.sum(time**2))/N
    E_xy = (np.sum(decay*time))/N
    m = (E_xy-E_x*E_y)/(E_xx-E_x**2)
    c = (E_xx*E_y-E_x*E_xy)/(E_xx-E_x**2)
    A = e**c
    t = c
    l = -m
    print('A =',A,'lambda =',l)
    x = np.linspace(0,3,100)
    y = m*x+c
    plt.plot(time,decay)
    plt.plot(x,y)
    A_m = np.outer(time.transpose(), decay)
    print(A_m)
    '''
    A = 
    q, r = np.linalg.qr(A, mode='r')
    func(t)
    '''
def func(t,A,l):
     return A*e**(-l*t)
 
def linear_function(t,A,l):
    return log(A)- l*t

uppgift3()

