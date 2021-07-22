import numpy as np
from scipy import linalg
import matplotlib.pyplot as mpl

### Paramètres du tuyau ###
L = 0.5
N = 10
umax = 15
du = 0.1


### Paramtètres adimensionnés ###
gamma = 0
beta = 0.2

### Fonction donnant les valeurs propres de poutre ###
def Eq_Cara(Lambda):
    return np.cos(Lambda)*np.cosh(Lambda)+1

LAMBDA = []

for i in range(N):
    LL_Guess = np.pi*(2*i+1)/2
    x0 = LL_Guess + 0.1
    x1 = LL_Guess - 0.1
    while abs(x0-x1)>10**-16:
        xnew = x0 - (x0-x1)*Eq_Cara(x0)/(Eq_Cara(x0)-Eq_Cara(x1))
        x1 = x0
        x0 = xnew
    LAMBDA.append(x0)

for i in LAMBDA:
    print("\nVecteur propre:")
    print(i)
    print("Valeur de l'équation:")
    print(Eq_Cara(i))
print("\n")


def sigma(r):
    return ((np.sinh(L*LAMBDA[r])-np.sin(L*LAMBDA[r]))/(np.cosh(L*LAMBDA[r])+np.cos(L*LAMBDA[r])))

def bsr(s,r):
    if s == r:
        return 2
    else:
        return 4/((LAMBDA[s]/LAMBDA[r])**2+(-1)**(r+s))
    
def csr(s,r):
    if s == r:
        return LAMBDA[r]*sigma(r)*(2-LAMBDA[r]*sigma(r))
    else:
        return 4*(LAMBDA[r]*sigma(r)-LAMBDA[s]*sigma(s))/((-1)**(r+s)-(LAMBDA[s]/LAMBDA[r])**2)
    
def dsr(s,r):
    if s == r:
        return csr(s,r)/2
    else:
        return (4*(LAMBDA[r]*sigma(r)-LAMBDA[s]*sigma(s)+2)*(-1)**(r+s))/(1-(LAMBDA[s]/LAMBDA[r])**4)-((3+(LAMBDA[s]/LAMBDA[r])**4)/(1-((LAMBDA[s]/LAMBDA[r])**4)))*bsr(s,r)                                                                                                                                               

def MAC(X,Y):
    return ((np.dot(X,np.conj(Y)).real)**2+(np.dot(X,np.conj(Y)).imag)**2)/(np.dot(X,np.conj(X))*np.dot(Y,np.conj(Y))).real



B = np.zeros((N,N))
C = np.zeros((N,N))
D = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

Delta = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4

MM = np.eye(N)


def result(u):
    C_g = 2*(beta**0.5)*u*B
    K = Delta + gamma*B + (u**2-gamma)*C + gamma*D
    F = np.block([[np.zeros((N,N)),MM],[MM,C_g]])
    E = np.block([[-MM,np.zeros((N,N))],[np.zeros((N,N)),K]])     
    eigenValues, eigenVectors = linalg.eig(-np.dot(np.linalg.inv(F),E))

    return eigenValues, eigenVectors
    
#def modes(u):

u_array = np.array([0])
u = 0
eigenValues, eigenVectors = result(u)
print(eigenValues)


arg = np.argsort(np.array([(-1j*eigenValues).real]))[0]

IM_Omega = np.array([(-1j*eigenValues).imag[arg]])
RE_Omega = np.array([(-1j*eigenValues).real[arg]])
Vectors = np.array([eigenVectors[:,arg]])


i = 0 
while u < umax:
    i += 1
    u_tempo = u + du  
    du_tempo = du
    
    eigenValues, eigenVectors = result(u_tempo)
    

    diff = np.zeros((2*N,2*N))
    for l in range(2*N):
        for k in range(2*N):
            diff[l,k] = MAC(Vectors[i-1,:,l],eigenVectors[:,k])
        
          
    
    while min(np.max(diff,1)) < 0.99 and du_tempo > 10**-10:
        
        u_tempo = u+du_tempo/2
        du_tempo = du_tempo/2
        eigenValues, eigenVectors = result(u)
        for l in range(2*N):
            for k in range(2*N):
                diff[l,k] = MAC(Vectors[i-1,:,l],eigenVectors[:,k])

    
    u = u_tempo
    u_array = np.append(u_array,u)
    arg_Max = np.argmax(diff,1) 
    IM_Omega = np.append(IM_Omega,[(-1j*eigenValues[arg_Max]).imag],axis=0)
    RE_Omega = np.append(RE_Omega,[(-1j*eigenValues[arg_Max]).real],axis=0)
    Vectors = np.append(Vectors,np.array([eigenVectors[:,arg_Max]]),axis=0)

   
   
mode1 = N
mode2 = N+1
mode3 = N+2
mode4 = N+3
   
mpl.plot(RE_Omega[:,mode1],IM_Omega[:,mode1],label="Mode 1")
mpl.plot(RE_Omega[:,mode2],IM_Omega[:,mode2],label="Mode 2")
mpl.plot(RE_Omega[:,mode3],IM_Omega[:,mode3],label="Mode 3")
mpl.plot(RE_Omega[:,mode4],IM_Omega[:,mode4],label="Mode 4")
mpl.xlabel("Re(Omega)")
mpl.ylabel("Im(Omega)")
mpl.title("Evolution de Omega en faisant varier la vitesse u")
mpl.legend()
mpl.show()

mpl.plot(u_array,RE_Omega[:,mode1],label="Mode 1")
mpl.plot(u_array,RE_Omega[:,mode2],label="Mode 2")
mpl.plot(u_array,RE_Omega[:,mode3],label="Mode 3")
mpl.plot(u_array,RE_Omega[:,mode4],label="Mode 4")
mpl.xlabel("u")
mpl.ylabel("Re(Omega)")
mpl.title("Evolution de la fréquence en faisant varier la vitesse u")
mpl.legend()
mpl.show()

mpl.plot(u_array,IM_Omega[:,mode1],label="Mode 1")
mpl.plot(u_array,IM_Omega[:,mode2],label="Mode 2")
mpl.plot(u_array,IM_Omega[:,mode3],label="Mode 3")
mpl.plot(u_array,IM_Omega[:,mode4],label="Mode 4")
mpl.xlabel("u")
mpl.ylabel("IM(Omega)")
mpl.title("Evolution de l'amortissement en faisant varier la vitesse u")
mpl.legend()
mpl.show()