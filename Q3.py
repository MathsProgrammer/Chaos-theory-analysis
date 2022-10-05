import numpy as np
import matplotlib.pyplot as plt

def pc2(f,x,p,dt):
    y  = f(x,p)
    xp = x + dt*y
    yp = f(xp,p)
    x1 = x + 0.5*dt*(y+yp)
    return x1



def rossler_f(x,p):
    f = np.empty(3)
    f[0] = -x[0]/p[0] + x[1] - x[1]*(x[0]**2 + x[1]**2)**0.5
    f[1] =  -2*x[1]/p[0] + x[0]*(x[0]**2 + x[1]**2)**0.5
    f[2] =  0
    return f




def rossler_map(x,p,dt):
    z_ = 1e9
    z__= 1e9
    for k in range(10000):
        if z__<z_ and z_>x[2]:
            return x
        z__= z_
        z_ = x[2]=0
        x  = pc2(rossler_f,x,p,dt)

####################################
dt = 0.01
T0 = 0.
T1 = 115.
Ns = int(T1/dt)
t  = np.linspace(T0,T1,Ns)
x  = np.empty((Ns,3))

p  = np.array([25])   # parameter
ep= [10.**(-6), 10.**(-5), 10.**(-4), 10.**(-3), 10.**(-2)]
for i in range(len(ep)):
    x[0,:] = np.array([0.,ep[i],0.])  # initial condition
    
    for q in range(Ns-1):
        x[q+1,:] = pc2(rossler_f,x[q,:],p,dt)
    
    plt.plot(t,np.log((x[:,0]**2+x[:,1]**2)**0.5), label="$\epsilon$ = " + str(ep[i]))
plt.legend()
plt.xlabel("time")
plt.ylabel("log($\sqrt{u_1^2+u_2^2})$")
plt.title("Log(speed) vs time for varied initial conditions")
plt.savefig("mod 98")
plt.show()
    

######### 
dt = 0.01
T0 = 0.
T1 = 50.
Ns = int(T1/dt)
t  = np.linspace(T0,T1,Ns)
x  = np.empty((Ns,3))

r  = np.array([2.5,4,50,10000])   # parameter
for i in range(len(r)):
    p=np.array([r[i]])
    x[0,:] = np.array([0.,0.1,0.])  # initial condition
    for q in range(Ns-1):
        x[q+1,:] = pc2(rossler_f,x[q,:],p,dt)
    
    plt.plot(t,np.log((x[:,0]**2+x[:,1]**2)**0.5), label="$R$ = " + str(r[i]))
plt.legend()
plt.xlabel("time")
plt.ylabel("$log(\sqrt{u_1^2+u_2^2})$")
plt.title("Log(speed) vs time for different R ")
plt.savefig("mod200")
plt.show()

