
import numpy as np
import matplotlib.pyplot as plt

def quadratic_gauss(N,a,b,function):
    '''
    kvadratisk gauss-metod som beräknar värdet av integralen av en funktion
    input: antal punkter, start- och slutpunkt för integralen och funktionen som 
    ska integreras
    '''
    # Calculate the sample points and weights, then map them
    # to the required integration domain
    x,w = gaussxw(N)
    xp = 0.5*(b-a)*x +0.5*(b+a)
    wp = 0.5*(b-a)*w
    # Perform the integration
    s=0.0
    for k in range(N):
        s+= wp[k]*function(xp[k])
    return s

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))
    # Find roots using Newton’s method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))
    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
    return x,w


def newtons(guess, f, df, iterations, p):
    x0 = guess
    for n in range(iterations):
        x0 = x0 -(f(x0))/(df(x0))
        if p == True:
            print(x0)
    return x0


