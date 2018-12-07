import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt,pi,e,tanh,cosh
 

#Uppgift 1

def func_upp_a(a,b,c):
    x_min = (-b+sqrt(b**2-4*a*c))/(2*a)
    x_max = (-b-sqrt(b**2-4*a*c))/(2*a)
    
    return x_min, x_max

def func_upp_b(a,b,c):
    x_min = (2*c)/(-b-sqrt(b**2-4*a*c))
    x_max = (2*c)/(-b+sqrt(b**2-4*a*c))
    
    return x_min, x_max

def func_upp_c(a,b,c):
    x_min = (2*c)/(-b-sqrt(b**2-4*a*c))
    x_max = (-b-sqrt(b**2-4*a*c))/(2*a)
    
    return x_min, x_max

#print('Uppgift a:',func_upp_a(0.001,1000,0.001),'\nUppgift b:', func_upp_b(0.001,1000,0.001))
#print('Uppgift c:', func_upp_c(0.001,1000,0.001))
#------------------------------------------------------------------------
    
#Uppgift 2

def f_1(x):
    return 0.02+1.01*x+0.1*x*x

def f_8(x):
    return 0.02+1.01*x+0.8*x*x


def c(f,c):
    mean,sigma1,sigma2=1,0.1,0.2
    x=mean+sigma1*np.random.randn(1000000)+sigma2*np.random.randn(1000000)
    df=f(x)-f(mean)
    
    
    x_new=np.linspace(-2,2,1000)
    sigma_x=sqrt(sigma1**2+sigma2**2)
    sigma_x1=(sigma_x**2)*(1.01+2*c)**2
    sigma=sqrt(sigma_x1)
    sigma_np = np.std(f(x))
    print('Calutated sigma:',sigma)
    print('With np.std :', sigma_np)
    y_new=(1/(sigma*sqrt(2*pi))*(e**(-(x_new)**2/(2*sigma**2))))
    
    plt.figure()
    plt.plot(x_new,y_new,color='red')
    
    n,bins,patches=plt.hist(df,100,density=1,facecolor='blue',label='Exact')
    
    plt.legend()
    plt.xlabel('f(x)-f(x_mean)')
    plt.ylabel('A.u.')
    plt.axis([-2, 2, 0, 2])
    plt.grid(True)
    plt.show()

    
#c(f_1,0.1)
#c(f_8,0.8)
#--------------------------------------------------------------------------
    
#Uppgift 3
ey=np.array([[0.0],[0.0],[0.001]])
y_vec=np.array([[1],[1],[1]]) 
M= np.matrix([[1.01,1.00,1.00],[1.00,1.01,1.00],[1.00,1.00,1.01]])
def condition_number_A(A):
    inv_A=np.linalg.inv(A)
    K=np.linalg.norm(A, np.inf)*np.linalg.norm(inv_A, np.inf)
    return K, inv_A

def e_f(ey,y,inv_A):
    taljare=np.linalg.norm(inv_A*ey, np.inf)/np.linalg.norm(inv_A*y_vec, np.inf)
    Namnare=np.linalg.norm(ey, np.inf)/np.linalg.norm(y_vec, np.inf)
    K = taljare/Namnare
    return K 


K, inv_A =condition_number_A(M)
K_annan =e_f(ey,y_vec,inv_A)
print(K)
print(np.linalg.norm(inv_A*ey, np.inf))
print(np.linalg.norm(inv_A*y_vec, np.inf))

#print('Enligt np.linalg.cond: ',np.linalg.cond(M, np.inf))
#print('Men funktionen e_f: ', K_annan)


#---------------------------------------------------------------------------

#uppgift 4
#5.6

def f(x):
    return x**4-2*x+1

def integral(N,a,b,f):
    h=(b-a)/N
    s=0.5*f(a)+0.5*f(b)
    for k in range(1,N):
        s+=f(a+(k*h))
    return h*s

def formel(integral):
    K= ((integral(20,0.0,2.0,f))-(integral(10,0.0,2.0,f)))/3
    return K

#print(integral(10,0.0,2.0,f))
#print(integral(20,0.0,2.0,f))
#print(formel(integral))
#print(integral(20,0.0,2.0,f)-4.4)

#5.10 a) i anteckningsboken
    #b)
def V(x):
    return x**4

def g(x,a,V):
    return sqrt(8)/sqrt(V(a)-V(x))
    
    
def integral_2(N,b,g):
    x,w = np.polynomial.legendre.leggauss(N)
    xp = 0.5*(b)*x +0.5*(b)
    wp = 0.5*(b)*w
    
    s=0.0
    for k in range(N):
        s+= wp[k]*g(xp[k],b,V)
    return s

def plot_pendel(a,b,N):
    x=np.linspace(a,b,N)
    y=[]
    for i in range(N):
        s=integral_2(20,x[i],g)
        y.append(s)
    plt.plot(x,y)

#plot_pendel(0.1,2.0,100)

#c) Vi har tittat på grafen och observerat att tidsintervallet blir kortare 
#   ju högre vi släpper bendeln 

#5.15
def func(x):
    return 1+(0.5*tanh(2*x))

def func_derivata(x):
    return 1/(cosh(2*x))**2

def dev_h(func, h, x ):
    return (func(x+h)-func(x-h))/(2*h)

def plot_der(a,b,N,h):
    x=np.linspace(a,b,N)
    y=dev_h(func,h,x)
    y_1=func_derivata(x)
    plt.plot(x,y_1,color='red', marker='_')
    plt.plot(x,y, color='blue' )
    
#plot_der(-2,2,100,0.1)
    
#------------------------------------------------------------------------
 
#Uppgift 5

def t(x):
    return tanh(x)
    
def ef(x,h):
    return 0.25*h**2*abs((-2)*tanh(x)*(1/cosh(x)**2))

def skattning(funk,h):
    a=0
    b=3/h
    y=[]
    for i in range(h):
        x=np.linspace(a,b,1000)
        s=funk(x)-(((funk(a)*(b-x))+((x-a)*funk(b)))/(b-a))
        s_max=max(s)
        y.append(s_max)
        a += 3/h
        b += 3/h
    maximum = max(y)
    return maximum

def egen_skatt(funk,a,b,h):
    h=3/h
    x=np.linspace(a,b,1000)
    y=funk(x,h)
    maximum = max(y)
    return maximum

#print('Med funktion skattning: ', skattning(t, 10))
#print('Med funktionen ef: ', egen_skatt(ef,0,3,10))


    
    
