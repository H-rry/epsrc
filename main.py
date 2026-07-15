import numpy as np 
import matplotlib as plt

length = 1
time = 100

dx = 0.1
dt = 0.05

Pe = 1

N = int(length // dx)

A = np.empty((N,N))
c_n = np.empty(N)
c_m = np.empty(N)


a = dt/(2*dx)
b = dt/((dx**2)*Pe)
beta = 1 + 2*a
alpha = a + b
gamma = b - a

def save():
    return 0

def fill_inital_condition(c):
    for i in range(N):
        c[i] = 0.0
    c[N-1] = 1
    return c

def fill_implicit_matrix(A):
    A.fill(0.0)
    
    A[0, 0] = 1.0
    

    for i in range(1, N - 1):
        A[i, i] = beta          
        A[i, i - 1] = -alpha
        A[i, i + 1] = -gamma

    A[N-1, N-1] = 1.0
    
    return A

def implicit(A, c_n, c_m):
    A = fill_implicit_matrix(A)
    t = 0
    while t <= time:
        print(c_n)
        c_m = np.linalg.solve(A,c_n)
        
        c_n = c_m
        t += dt



def main():
    implicit(A, fill_inital_condition(c_n), c_m)


if __name__ == "__main__":
    main()