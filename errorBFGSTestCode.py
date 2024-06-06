
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pycutest

# Build this problem with N=300, M=100
ROSENBR = pycutest.import_problem('DIXMAANH', sifParams={ 'M':100})
np.random.seed(69)
firstX = ROSENBR.x0
print(firstX.shape)
f, gradSize = ROSENBR.obj(firstX, gradient=True)  # objective and gradient


def fobj(ai, func):
    return func.obj(ai) + np.random.uniform(-0.001,0.001)

def gobj(ai, func):
    _, grad = func.obj(ai, gradient=True)
    return grad + np.random.uniform(-0.001,0.001)

def trueF(ai, func):
    return func.obj(ai)

def trueG(ai, func):
    _, grad = func.obj(ai, gradient=True)
    return grad





def SplitPhase(func, eg, x, p, a, B, f_k, g_k, noiseCon, c1 = 1e-4, c2 = 0.9, c3 = 0.5):
    
   
    #Armijo Condition
    while fobj(x + a*p, func) > f_k + c1*a*g_k.T@p:

        #Backtrack
        a = a/10

    #Noise Control Condition
    while (gobj(x+ B*p,func) - g_k).T @ p < noiseCon:

        #Lengthen
        B = 2*B

    return a, B
    

def relaxedArmijo( p, i, a, f_k, g_k, f_ap, eg=0, ef=0, c1 = 1e-4):
    
    
    # Calculate the gradient descent step
    descent_step = a * np.dot(g_k, p)
    
    if descent_step < -eg * np.linalg.norm(p):
        # Check Armijo condition for sufficient decrease
        if i == 0:
            if f_ap > (f_k + c1 * a * descent_step):
                return True
        else:
            if f_ap > (f_k + c1 * a * descent_step) + 2 * ef:
                return True
    else:
        # Check for non-sufficient decrease
        if i == 0:
            if f_ap >= f_k:
                return True
        else:
            if f_ap >= f_k + 2 * ef:
                return True
    
    # If neither condition satisfies, return False
    return False

    

def initialPhaseLineSearch(func, eg, ef, x, p, f_k, g_k, mu_history, c1 = 1e-4, c2 = 0.9, c3 = 0.5, Nsplit = 30):
    # Initializing brackets for bisection
    a = 1.
    abest = 1
    fbest = np.inf
    l = 0
    u = np.inf
    B = 1.
    noiseControl = 2*(1+c3)*eg*np.linalg.norm(p)


    for i in range(Nsplit):
        f_ap = fobj(x + a*p,func)
        g_ap = gobj(x+a*p, func)
        if relaxedArmijo( p, i, a, f_k, g_k, f_ap, eg = eg, ef = ef):
            u = a

            #Backtrack
            a = (a+l)/2.0
            if f_ap < fbest:
                abest = a
                fbest = f_ap
            B = a
        
        #Noise Control
        elif np.abs(np.dot(g_ap-g_k,p)) < noiseControl:
          
            break
            
        #Wolfe Condition fails
        elif np.dot(g_ap,p) < c2*np.dot(g_k,p):
            
            l = a

            #Advance
            if u == np.inf:
                a = 2.0*a
                if f_ap < fbest:
                    abest = a
                    fbest = f_ap
            else:
                a = (u+l)/2.0
                if f_ap < fbest:
                    abest = a
                    fbest = f_ap
                B = a
        
        #Satisfies all conditions
        else:
            B = a
           
            return a, B
    
    a = abest
    if len(mu_history != 0):
        B = np.maximum(2*B, (2*(1+c3)*eg)/(np.min(mu_history)*np.linalg.norm(p)))
    a, B = SplitPhase(func, eg, x, p, a, B,f_k,g_k, noiseControl)
    return a,B


def NoiseTolerantBFGS(max_iter, func, x0, H, n, ef = 0, eg = 0, c1 = 1e-4, c2 = 0.9, c3 = 0.5):
    IMat = np.eye(gradSize.shape[0])
    Hk = H
    xk = x0
    mu_hist = np.array([])
    for k in range(max_iter):
        g_k = gobj(xk, func)
        f_k = fobj(xk,func)

        pk = np.dot(-Hk,g_k)
        
        alpha, Beta = initialPhaseLineSearch(func, eg, ef, xk, pk, f_k, g_k, mu_hist)
        
        #Satisfying inequality 4.1 from the paper
        if fobj(xk + alpha*pk, func) <= f_k + (c1*alpha)*np.dot(g_k,pk):
            xk1 = xk + alpha*pk
        
        #Checking that beta satisfies inequality 2.9 from paper
        g_bp = gobj(xk+ Beta*pk,func)
        if np.dot(g_bp - g_k, pk) >= 2*(1+c3)*eg*np.linalg.norm(pk):
            #Computing curvature pairs sk, yk
           
            sk = Beta*pk
            yk = g_bp - g_k
            
            np.append(mu_hist, np.dot(g_bp,pk)/(Beta*(np.linalg.norm(pk)**2)))
            if mu_hist.shape[0] > 10:
                    mu_hist = mu_hist[1:]
            #Updating Hk via BFGS formula 
            #TODO: Implement L-BFGS Version
            # heuristic at first iteration to capture scaling
            
            if k == 0:
                Hk = np.dot(yk, sk) / np.dot(yk, yk) * np.identity(n)

            # update BFGS matrix
            rho = 1. / np.dot(sk, yk)
            
            mat = np.identity(n) - rho * np.outer(sk, yk)
            
            Hk = np.matmul(np.matmul(mat, Hk), mat.transpose()) + rho * np.outer(sk, sk)
            xk = np.copy(xk1)
    '''print("Value of x_k:")
    print(xk)
    print("Gradient Norm:")
    print(np.linalg.norm(trueG(xk,func)))
    print("Objective Func Value:")
    print(trueF(xk, func))'''
    
    return trueF(xk, func)



sum = NoiseTolerantBFGS(1000,ROSENBR, firstX, np.eye(gradSize.shape[0]), len(firstX), ef = 0, eg = 0)

print(sum)


'''avgFNoE = 0
avgFE = 0
avgGE = 0
avgGNoE = 0
for i in range(10):
    
    fnoe, gnoe = NoiseTolerantBFGS(1000,ROSENBR, firstX, np.eye(gradSize.shape[0]), len(firstX))
    fe, ge = NoiseTolerantBFGS(1000,ROSENBR, firstX, np.eye(gradSize.shape[0]), len(firstX), ef = 0.001, eg = (np.sqrt(300)*0.001))
    avgFNoE += fnoe
    avgFE += fe
    avgGE += ge
    avgGNoE = gnoe


print("Average Objective Reg BFGS:")
print(avgFNoE/10)
print("Average Norm Reg BFGS:")
print(avgGNoE/10)
print("--------------------------")
print("Average Objective E-BFGS:")
print(avgFE/10)
print("Average Norm E-BFGS:")
print(avgGE/10)'''

   






    



















