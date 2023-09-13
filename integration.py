import numpy as np

""" Definicion de las funciones """

def Trapezoid(funk, a, b, N,*args, Romberg=False):
    h = (b-a)/N
    I0 = (h/2)*(funk(a,args[0],args[1],args[2])+funk(b,args[0],args[1],args[2]))
    Fc , x0 = 0 , a
    for i in range(1,N):
        xi = x0+(i*h)
        Fc+= funk(xi,args[0],args[1],args[2])
    Ic = I0 +(h*Fc)
    
    if Romberg == False:
        Nprime = int(N/2)
        hp = (b-a)/Nprime
        I0p = (hp/2)*(funk(a,args[0],args[1],args[2])+funk(b,args[0],args[1],args[2]))
        Fcp , x0p = 0 , a
        for i in range(1,Nprime):
            xi = x0p+(i*hp)
            Fcp+= funk(xi,args[0],args[1],args[2])

        Icp = I0p +(hp*Fcp)
        error = (1/3)*(Ic-Icp)

        return Ic, error
    else:
        return Ic
    
def Simpson(funk, a, b, N,*args):
    if (N+1)%2 != 0:
        print('Error. N should be odd for this method')
    else:
        h = (b-a)/N
        I0 = (h/3)*(funk(a,args[0],args[1],args[2])+funk(b,args[0],args[1],args[2]))
        Fc , x0 = 0 , a
        for i in range(1,N-1,2):
            Fc+= 4*funk(x0+(i*h),args[0],args[1],args[2]) + 2*funk(x0+(2*i*h),args[0],args[1],args[2])
        Ic = I0 +(h*Fc)/3
        
        Nprime = int(N/2)
        hp = (b-a)/Nprime
        I0p = (hp/3)*(funk(a,args[0],args[1],args[2])+funk(b,args[0],args[1],args[2]))
        Fcp , x0p = 0 , a
        for i in range(1,Nprime-1,2):
            Fcp+= 4*funk(x0p+(i*hp),args[0],args[1],args[2]) + 2*funk(x0p+(2*i*hp),args[0],args[1],args[2])
        Icp = I0p +(hp*Fcp)/3

        error = (1/15)*(Ic-Icp)
        
        return Ic, error

def Romberg(funk, a, b, N,*args):
    R = np.zeros((N,N),float)
    for i in range(N):
        R[i][0] = Trapezoid(funk, a, b, (2**i), args[0],args[1],args[2],Romberg=True)
        for j in range(1,N):    
            if j <= i:
                num = ((4**j)*R[i][j-1]) -(R[i-1][j-1]) 
                den = (4**j)-1
                R[i][j] =  num/den
            else: 
                pass
    I_R = R[-1][-1]
    h   = (b-a)/(N) 
    er  = R[-1][-1] - R[-1][-2]
        
    return I_R, er

