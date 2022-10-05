import numpy as np
import matplotlib.pyplot as plt

#Q1
#############################################################
def invmap(x,r):
    return r*x/(1.+x*x)

rval = [0.5,1,1.5,2,5,17]

Np = 50
xp = np.linspace(0,8,Np)    
fp = np.empty(Np)            
for j in rval:
    for i in range(Np):
        fp[i] = invmap(xp[i],j) 
    plt.xlabel("$x_t$")
    plt.ylabel("$x_{t+1}$")    
    plt.plot(xp,xp,"g-") 
    plt.plot(xp,fp,label="r="+str(j)) 
plt.legend()
plt.savefig("M1.png")
plt.show() 

######################################################################
f
rval = [0.5,1,1.5,2,5,17] 
Np = 50
tp = np.arange(Np)  
xp = np.empty(Np) 
xp[0] = 6
for j in rval:
    for i in range(Np-1):
        xp[i+1] = invmap(xp[i],j)
    plt.xlabel("Time")
    plt.ylabel("Population of lemmings")
    plt.plot(tp,xp,label="r="+str(j))
plt.legend()
plt.savefig("M2")
plt.show()

#######################################################################

def lemdif(x,r):
    return r*(1-x**2)/((1.+x*x)**2)

Nr = 120
rr = np.linspace(0,2,Nr)
ly = np.empty(Nr)
x  = 6
 
for j in range(Nr):
    r = rr[j]
    A = 0.
    for k in range(1000):
        A = A + np.log(abs(lemdif(x,r)))
        x = invmap(x,r)
    ly[j] = A / 1000

zz = np.zeros(Nr)
plt.plot(rr,zz,'b-')
plt.plot(rr,ly,'r.')
plt.title("Lyapunov exponent for r parameter")
plt.ylabel("Lyapunov exponent")
plt.xlabel("r parameter")
plt.savefig("lya lem")
plt.show()

#########################################################################

Nr = 225
Np = 100
rr = np.linspace(0,500,Nr)
pr = np.empty(Nr*Np)
px = np.empty(Nr*Np)
i  = 0

for r in rr:
    x = 0.5
    for k in range(100):      #iterate to get to attractor
        x = invmap(x,r)
    for k in range(Np):       #iterates to plot
        x = invmap(x,r)
        pr[i] = r
        px[i] = x
        i += 1

plt.plot(pr,px,'b,',markersize=0.11,alpha = 1)
plt.xlabel("r parameter")
plt.ylabel("lemming population")
plt.title("Bifurcation plot")
plt.savefig("bifur lem")
plt.show()

#########################################################################

def electricalCurrent1(x,r):
    return np.exp(-r*x)
 
rval = [0.2,1,2.5,8]
alpha = [1,1,1,0.3]
#rval=[2.7182]
Np = 10000
tp = np.arange(Np)  #array from 0-np, equally spaced. eg 1,2,3 -> 20
xp = np.empty(Np) 
xp[0] = 0.5
for j in range(len(rval)):
    for i in range(Np-1):
        xp[i+1] = electricalCurrent1(xp[i],rval[j])
    plt.xlabel("Time")
    plt.ylabel("Electric current")
    plt.plot(tp[0:100],xp[0:100],label="r="+str(rval[j]),alpha = alpha[j])
plt.legend(loc="upper right")  
#plt.savefig("mod5")  
plt.show()

def electricalCurrent1(x,r):
    return np.exp(-r*x)
 
#rval = [0.2,1,2.5,8]

rval=2.5
Np = 10000
tp = np.arange(Np)  #array from 0-np, equally spaced. eg 1,2,3 -> 20
xp = np.empty(Np) 
xs = np.array([0,50,200])
alpha = [0.8,0.8,0.8]
for j in range(len(xs)):
    for i in range(Np-1):
        xp[0] = xs[j]
        xp[i+1] = electricalCurrent1(xp[i],rval)
    plt.xlabel("Time")
    plt.ylabel("Electrical currents")
    plt.plot(tp[0:50],xp[0:50],label="x(0)="+str(xs[j]),alpha = alpha[j])
plt.legend(loc="upper right")  
plt.ylim(0,1)
plt.savefig("mod5")  
plt.show()

#print(xp[9900000],xp[9900001])
#print(xp[9000],xp[9001])


rval = [0.2,1,2,2.5,8]
Np = 50
xp = np.linspace(0,4,Np)     
fp = np.empty(Np)            
for j in rval:
    for i in range(Np):
        fp[i] = electricalCurrent1(xp[i],j)  
    plt.xlabel("$x_t$")
    plt.ylabel("$x_{t+1}$")    
    plt.plot(xp,xp,"g-") 
    plt.plot(xp,fp,label="r="+str(j))
plt.legend()
plt.savefig("mod1")
plt.show()    
   
## bifurcation
Nr = 500
Np = 100
rr = np.linspace(0,5,Nr)
pr = np.empty(Nr*Np)
px = np.empty(Nr*Np)
i  = 0

for r in rr:
    x = 10**(-6)
    for k in range(100):      #iterate to get to attractor
        x = electricalCurrent1(x,r)
    for k in range(Np):       #iterates to plot
        x = electricalCurrent1(x,r)
        pr[i] = r
        px[i] = x
        i += 1

plt.plot(pr,px,'b.',markersize=0.25,alpha = 1)
plt.xlabel("Input voltage")
plt.ylabel("Current")
plt.savefig("mod3")
plt.show()    

#lyapunov
def e1dif(x,r):
    return -r*np.exp(-r*x)

Nr = 120
rr = np.linspace(0,10,Nr)
ly = np.empty(Nr)
x  = 8
 
for j in range(Nr):
    r = rr[j]
    A = 0.
    for k in range(1000):
        A = A + np.log(abs(e1dif(x,r)))
        x = electricalCurrent1(x,r)
    ly[j] = A / 1000

zz = np.zeros(Nr)
plt.plot(rr,zz,'b-')
plt.plot(rr,ly,'r.')
plt.title("Lyapunov exponent for voltage")
plt.ylabel("Lyapunov exponent")
plt.xlabel("Input voltage")
plt.savefig("mod4")
plt.show()


def electricalCurrent2(x,r):
    return r*np.cos(x)
 
rval = [0.2,1,2.5,8]
alpha = [1,1,1,0.3]
#rval=[2.7182]
Np = 10000
tp = np.arange(Np)  #array from 0-np, equally spaced. eg 1,2,3 -> 20
xp = np.empty(Np) 
xp[0] = 0.5
for j in range(len(rval)):
    for i in range(Np-1):
        xp[i+1] = electricalCurrent2(xp[i],rval[j])
    plt.xlabel("Time")
    plt.ylabel("Electrical currents")
    plt.plot(tp[0:100],xp[0:100],label="r="+str(rval[j]),alpha = alpha[j])
plt.legend(loc="upper right")  
plt.savefig("mod11")  
plt.show()

rval=1.4
Np = 10000
tp = np.arange(Np)  #array from 0-np, equally spaced. eg 1,2,3 -> 20
xp = np.empty(Np) 
xs = np.array([0,10,200])
alpha = [0.8,0.8,0.8]
for j in range(len(xs)):
    for i in range(Np-1):
        xp[0] = xs[j]
        xp[i+1] = electricalCurrent2(xp[i],rval)
    plt.xlabel("Time")
    plt.ylabel("Electrical currents")
    plt.plot(tp[0:10000],xp[0:10000],label="x(0)="+str(xs[j]),alpha = alpha[j])
plt.legend(loc="upper right")  
plt.ylim(0.2,1.5)
plt.savefig("mod12")  
plt.show()

#bifurcation
Nr = 250
Np = 100
rr = np.linspace(0,10,Nr)
pr = np.empty(Nr*Np)
px = np.empty(Nr*Np)
i  = 0

for r in rr:
    x = 10
    for k in range(100):      #iterate to get to attractor
        x = electricalCurrent2(x,r)
    for k in range(Np):       #iterates to plot
        x = electricalCurrent2(x,r)
        pr[i] = r
        px[i] = x
        i += 1

plt.plot(pr,px,'b.',markersize=0.15,alpha = 1)
plt.xlabel("Input voltage")
plt.ylabel("Current")
plt.savefig("mod9")
plt.show()   

#lyaponov

def e2dif(x,r):
    return -r*np.sin(x)

Nr = 1200
rr = np.linspace(0,10,Nr)
ly = np.empty(Nr)
x  = 8
 
for j in range(Nr):
    r = rr[j]
    A = 0.
    for k in range(1000):
        A = A + np.log(abs(e2dif(x,r)))
        x = electricalCurrent2(x,r)
    ly[j] = A / 1000

zz = np.zeros(Nr)
plt.plot(rr,zz,'b-')
plt.plot(rr,ly,'r.')
plt.title("Lyapunov exponent for voltage")
plt.ylabel("Lyapunov exponent")
plt.xlabel("Input voltage")
plt.savefig("mod10")
plt.show()  

#Q2
###################################################################

def tinkerbell(x,p):
    x1 = np.empty(2)
    x1[0] = x[0]**2 -x[1]**2 +p[0]*x[0] +p[1]*x[1]
    x1[1] = 2.*x[0]*x[1] + p[2]*x[0] + p[3]*x[1]
    return x1
Np = 5000
p = np.array([-0.3,-0.6,2.,0.5])
x = np.empty((Np,2))
x[0,:] = np.array([0.1,0.1])
for i in range(Np-1):
    x[i+1,:] = tinkerbell(x[i,:],p)
plt.plot(x[1000:1070,0],x[1000:1070,1], "r-")
plt.xlabel("$x_n$")
plt.ylabel("$y_n$")
plt.savefig("mod90")
#It's a repeating pattern, no point going over 1000

#Mess about with c for answer
#-.2 lowest 1.2 high
#B
def tinkerbellC(x,p):
    x1 = np.empty(2)
    x1[0] = x[0]**2 -x[1]**2 +p[0]*x[0] +p[1]*x[1]
    x1[1] = 2.*x[0]*x[1] + p[2]*x[0] + p[3]*x[1]
    return x1
c = 0.4
Np = 5000
p = np.array([-0.3,-0.6,2.,c])
x = np.empty((Np,2))
x[0,:] = np.array([0.1,0.1])
for i in range(Np-1):
    x[i+1,:] = tinkerbellC(x[i,:],p)
plt.plot(x[1000:,0],x[1000:,1], "r.")
plt.xlabel("$x_n$")
plt.ylabel("$y_n$") 
plt.savefig("mod30") 
plt.show()  

def tinkerbellC(x,p):
    x1 = np.empty(2)
    x1[0] = x[0]**2 -x[1]**2 +p[0]*x[0] +p[1]*x[1]
    x1[1] = 2.*x[0]*x[1] + p[2]*x[0] + p[3]*x[1]
    return x1
c = 0.7
Np = 5000
p = np.array([-0.3,-0.6,2.,c])
x = np.empty((Np,2))
x[0,:] = np.array([0.1,0.1])
for i in range(Np-1):
    x[i+1,:] = tinkerbellC(x[i,:],p)
plt.plot(x[1000:,0],x[1000:,1], "r.")
plt.xlabel("$x_n$")
plt.ylabel("$y_n$")
plt.savefig("mod31")  
plt.show()

#Q3
###########################################################
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

r  = np.array([2.5,4,50,10000])   
for i in range(len(r)):
    p=np.array([r[i]])
    x[0,:] = np.array([0.,0.1,0.])  
    for q in range(Ns-1):
        x[q+1,:] = pc2(rossler_f,x[q,:],p,dt)
    
    plt.plot(t,np.log((x[:,0]**2+x[:,1]**2)**0.5), label="$R$ = " + str(r[i]))
plt.legend()
plt.xlabel("time")
plt.ylabel("$log(\sqrt{u_1^2+u_2^2})$")
plt.title("Log(speed) vs time for different R ")
plt.savefig("mod200")
plt.show()