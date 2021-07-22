import numpy as np
from scipy import linalg
import matplotlib.pyplot as mpl


### Paramètres du tuyau ###
L = 0.5
E = 0.06895*10**6
#E = 10**6
d = 6.35*10**-3
D = 15.875*10**-3
I = np.pi*(D**4-d**4)/64
rho_eco = 1064.61
rho_eau = 1000
M = rho_eau*d**2/4
m = rho_eco*(D**2-d**2)/4
g = 9.81
N = 10

### Paramtètres adimensionnés ###
beta = M/(M+m)
gamma = (m+M)*L**3*g/(E*I)
array_beta = np.linspace(0,1,100)
array_gamma = np.array([-10,0,10,100])

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

U = np.linspace(0,4,40)
u_array = (M/(E*I))**0.5*L*U
u_array = np.linspace(0,50,1000)


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

u_critique = np.zeros((len(array_gamma),len(array_beta)))

for g in range(len(array_gamma)):
    for b in range(len(array_beta)):
        beta = array_beta[b]
        gamma = array_gamma[g]
        u = u_array[0]
        MM = np.eye(N)
        C_g = 2*beta**0.5*u*B
        K = Delta + gamma*B + (u**2-gamma)*C + gamma*D
        F = np.block([[np.zeros((N,N)),MM],[MM,C_g]])
        E = np.block([[-MM,np.zeros((N,N))],[np.zeros((N,N)),K]])     
        
        eigenValues, eigenVectors = linalg.eig(-np.dot(np.linalg.inv(F),E))
        
        # if min((-1j*eigenValues).imag) < -0.0001:
        #     print("Oui")
        #     u_critique[g,b] = u
        
        for i in range(1,len(u_array)):
            u = u_array[i]
            MM = np.eye(N)
            C_g = 2*beta**0.5*u*B
            K = Delta + gamma*B + (u**2-gamma)*C + gamma*D
            F = np.block([[np.zeros((N,N)),MM],[MM,C_g]])
            E = np.block([[-MM,np.zeros((N,N))],[np.zeros((N,N)),K]])     
        
            eigenValues, eigenVectors = linalg.eig(-np.dot(np.linalg.inv(F),E))
            
            Arg = np.argmin((-1j*eigenValues).imag)
                
            if (-1j*eigenValues).imag[Arg] < -0.1 and u_critique[g,b] == 0 and (-1j*eigenValues).real[Arg] != 0:
                u_critique[g,b] = u
                
mpl.plot(array_beta,u_critique[0,:],label='Gamma = -10')
mpl.plot(array_beta,u_critique[1,:],label='Gamma = 0')
mpl.plot(array_beta,u_critique[2,:],label='Gamma = 10')
mpl.plot(array_beta,u_critique[3,:],label='Gamma = 100')
mpl.legend()
mpl.xlabel("Beta")
mpl.ylabel("U critique")
mpl.title("Flutter instability")
mpl.savefig('flutter.jpg')
mpl.show()
