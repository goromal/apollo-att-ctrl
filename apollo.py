import numpy as np
from geometry import SO3

class ApolloParameters:
    J = np.array([[ 40823.16316, -1537.807585,  3179.297464],
                  [-1537.807585,  90593.60312, -128.5771214],
                  [ 3179.297464, -128.5771214,  98742.98457]])
    J_simp = np.array([[40482.0736,          0,             0],
                       [         0, 90358.4306,             0],
                       [         0,          0,    98637.0675]])

class ApolloState:
    def __init__(self, q=SO3.identity(), w=np.zeros((3,1))):
        self.q = q
        self.w = w
    
    def __add__(self, v):
        return ApolloState(self.q + v[:3,0:1], self.w + v[3:,0:1])
        
    def __sub__(self, other):
        return np.vstack((self.q - other.q, self.w - other.w))

def ApolloDynamics(x, u, simple=False):
    J = ApolloParameters.J_simp if simple else ApolloParameters.J
    J_inv = np.linalg.inv(J)
    
    q_dot = x.w
    w_dot = J_inv.dot(np.cross(-x.w, J.dot(x.w), axis=0) + u)

    return np.vstack((q_dot, w_dot))
    
def ApolloController(k_q, k_w, x, q_d, w_d=np.zeros((3,1)), wdot_d=np.zeros((3,1)), euler_error=False, feedback_lin=True, simple=False):
    J = ApolloParameters.J_simp if simple else ApolloParameters.J
    R = x.q.R()
    R_d = q_d.R()
    w = x.w
    
    if euler_error:
        r, p, y = x.q.toEuler()
        r_d, p_d, y_d = q_d.toEuler()
        e_q = np.array([[r-r_d],[p-p_d],[y-y_d]])
    else:
        e_q = x.q - q_d
    e_w = w - R.T.dot(R_d.dot(w_d))
    
    u = -k_q * e_q - k_w * e_w
    
    if feedback_lin:
        u += np.cross(w, J.dot(w), axis=0)
        
        if not np.linalg.norm(w_d) == 0:
            u += -J.dot(SO3.hat(w).dot(R.T.dot(R_d.dot(w_d))))
            
        if not np.linalg.norm(wdot_d) == 0:
            u += J.dot(R.T.dot(R_d.dot(wdot_d)))
    
    return u