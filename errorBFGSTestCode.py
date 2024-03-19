

import numpy as np
import pycutest

ROSENBR = pycutest.import_problem('WATSON')
firstX = ROSENBR.x0
f, gradSize = ROSENBR.obj(firstX, gradient=True)  # objective and gradient


def fobj(ai, func):
    return func.obj(ai)

def gobj(ai, func):
    _, grad = func.obj(ai, gradient=True)
    return grad





def SplitPhase(func, eg, x, p, a, B,c1 = 1e-4, c2 = 0.9, c3 = 0.5):
    

    #Armijo Condition
    while fobj(x + a*p, func) > fobj(x, func) + c1*a*gobj(x,func).T@p:

        #Backtrack
        a = a/10

    #Noise Control Condition
    while (gobj(x+ B*p,func) - gobj(x,func)).T @ p < 2*(1+c3)*eg*np.linalg.norm(p):

        #Lengthen
        B = 2*B

    return a, B
    


def betterInitial(func, eg, ef, x_k, p_k, c1 = 1e-4, c2 = 0.9, c3 = 0.5, Nsplit = 30):
    # initialize line search
    armijo_flag = False
    wolfe_flag = False
    split_flag = False
    # define upper and lower brackets
    upper_bracket = np.inf
    lower_bracket = 0

    # compute f_k and g_k if necessary
    
    f_k = fobj(x_k,func)
    g_k = gobj(x_k,func)


    eps_fk = np.copy(ef)
    
    eps_gk = np.copy(eg)

    # compute dot product and norm
    gtp = np.dot(g_k, p_k)
    norm_pk = np.linalg.norm(p_k)

    # initialize quantities
    g_new = np.inf
    gtp_new = np.inf

    # define both steplength and lengthening parameter
    alpha = 1
    beta = 1

    # store best steplength, strong convexity parameter
    alpha_best = 0.
    f_best = np.inf
    eps_fbest = np.inf
    rhs = None

    # check if steplength and lengthening parameter are different
    if alpha != beta:
        split_flag = True

        # evaluate gradient
        g_new = grad(x_k + beta * p_k)
        grad_evals += 1
        gtp_new = np.dot(g_new, p_k)




def initialPhaseLineSearch(func, eg, ef, x, p, c1 = 1e-4, c2 = 0.9, c3 = 0.5, Nsplit = 30):
    # Initializing brackets for bisection
    a = 1.
    l = 0
    u = np.inf
    B = 1.



    for i in range(Nsplit):
        
        #Arimijo condition failure check
        if fobj(x + a*p, func) > (fobj(x,func) + c1*a*np.dot(gobj(x,func),p)):
            
            u = a

            #Backtrack
            a = (a+l)/2.0
            B = a

        
        #Noise Control
        elif np.abs(np.dot(gobj(x+a*p, func)-gobj(x, func),p)) < 2*(1+c3)*eg*np.linalg.norm(p):
          
            break
            
        #Wolfe Condition fails
        elif np.dot(gobj(x + a*p,func),p) < c2*np.dot(gobj(x,func),p):
            
            l = a

            #Advance
            if u == np.inf:
                a = 2.0*a
            else:
                a = (u+l)/2.0
                B = a
        
        #Satisfies all conditions
        else:
            B = a
           
            return a, B
    
    a, B = SplitPhase(func, eg, x, p, a, B)
    return a,B


def NoiseTolerantBFGS(func, x0, H, n, ef = 0, eg = 0, c1 = 1e-4, c2 = 0.9, c3 = 0.5):
    IMat = np.eye(gradSize.shape[0])
    Hk = H
    xk = x0

    
    for k in range(100):
        
        pk = np.dot(-Hk,gobj(xk, func))
        
        alpha, Beta = initialPhaseLineSearch(func, eg, ef, xk, pk)
        
        #Satisfying inequality 4.1 from the paper
        if fobj(xk + alpha*pk, func) <= fobj(xk,func) + (c1*alpha)*np.dot(gobj(xk,func),pk):
            xk1 = xk + alpha*pk
        
        #Checking that beta satisfies inequality 2.9 from paper
        if np.dot(gobj(xk+ Beta*pk,func) - gobj(xk,func), pk) >= 2*(1+c3)*eg*np.linalg.norm(pk):
            #Computing curvature pairs sk, yk
           
            sk = Beta*pk
            yk = gobj(xk+ Beta*pk,func) - gobj(xk,func)
            
        
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
        
            
    print(xk)
            




NoiseTolerantBFGS(ROSENBR, firstX, np.eye(gradSize.shape[0]), len(firstX))



    



















