import numpy as np
import matplotlib.pyplot as plt



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