
import numpy as np
import matplotlib.pyplot as plt
from geometry import SO3
from signals import integrateRK4, timeDerivative
from apollo import ApolloState, ApolloDynamics, ApolloController

commandSettings = {
    'ATT': {'w_d': lambda t, q : [np.zeros((3,1)) for i in range(len(q))],
            'wdot_d': lambda t, q : [np.zeros((3,1)) for i in range(len(q))]},
    'RATE': {'w_d': lambda t, q : timeDerivative(t, q, tangent_dim=3),
             'wdot_d': lambda t, q : [np.zeros((3,1)) for i in range(len(q))]},
    'FULL': {'w_d': lambda t, q : timeDerivative(t, q, tangent_dim=3),
             'wdot_d': lambda t, q : timeDerivative(t, timeDerivative(t, q, tangent_dim=3))}
}

controllerSettings = {
    'EULER_LINEAR_SIMPLE': {'euler_error': True, 
                            'feedback_lin': False, 
                            'simple': True,
                            'k_q': 50000.0,
                            'k_w': 100000.0},
    'EULER_LINEAR_FULL': {'euler_error': True, 
                          'feedback_lin': False, 
                          'simple': False,
                          'k_q': 1.0,
                          'k_w': 1.0},
    'EULER_NONLIN_SIMPLE': {'euler_error': True, 
                            'feedback_lin': True, 
                            'simple': True,
                            'k_q': 50000.0,
                            'k_w': 100000.0},
    'EULER_NONLIN_FULL': {'euler_error': True, 
                          'feedback_lin': True, 
                          'simple': False,
                          'k_q': 1.0,
                          'k_w': 1.0},
    'MANIF_LINEAR_SIMPLE': {'euler_error': False, 
                            'feedback_lin': False, 
                            'simple': True,
                            'k_q': 50000.0,
                            'k_w': 100000.0},
    'MANIF_LINEAR_FULL': {'euler_error': False, 
                          'feedback_lin': False, 
                          'simple': False,
                          'k_q': 1.0,
                          'k_w': 1.0},
    'MANIF_NONLIN_SIMPLE': {'euler_error': False, 
                            'feedback_lin': True, 
                            'simple': True,
                            'k_q': 50000.0,
                            'k_w': 100000.0},
    'MANIF_NONLIN_FULL': {'euler_error': False, 
                          'feedback_lin': True, 
                          'simple': False,
                          'k_q': 1.0,
                          'k_w': 1.0}
}

##################################################
comType = 'FULL'
conType = 'MANIF_NONLIN_SIMPLE'
comStep = True
##################################################

comSet = commandSettings[comType]
conSet = controllerSettings[conType]

t0, dt, tf = 0, 0.02, 20

t = np.arange(t0, tf, dt)
n = t.size

# q_d_step = [SO3.fromEuler(np.pi/2, 0, 0)] * n
# q_d_step = [SO3.fromEuler(0, np.pi/4, 0)] * n
# q_d_step = [SO3.fromEuler(0, np.pi/2, 0)] * n
q_d_step = [SO3.fromEuler(0.1, -0.1, 0.2)] * n
# q_d_step = [SO3.fromEuler(np.pi/2, np.pi/4, -np.pi/3)] * n
# q_d_step = [SO3.fromEuler(np.pi/2, np.pi, np.pi/2)] * n

q_d_cont = [SO3.identity()] * n # list()
for i in range(n):
    pass # TODO
    
w_d_step = comSet['w_d'](t, q_d_step)
wdot_d_step = comSet['wdot_d'](t, q_d_step)
w_d_cont = comSet['w_d'](t, q_d_cont)
wdot_d_cont = comSet['wdot_d'](t, q_d_cont)

if comStep:
    q_d, w_d, wdot_d = q_d_step, w_d_step, wdot_d_step
else:
    q_d, w_d, wdot_d = q_d_cont, w_d_cont, wdot_d_cont

x = ApolloState()

psi  = np.empty(n)
tht  = np.empty(n)
phi  = np.empty(n)
psi_c = np.empty(n)
tht_c = np.empty(n)
phi_c = np.empty(n)

def dynamics(x, u):
    global conSet
    return ApolloDynamics(x, u, conSet['simple'])

for i in range(n):
    phi[i], tht[i], psi[i] = x.q.toEuler()
    phi_c[i], tht_c[i], psi_c[i] = q_d[i].toEuler()
    
    u = ApolloController(conSet['k_q'], 
                         conSet['k_w'], 
                         x, 
                         q_d[i], 
                         w_d[i], 
                         wdot_d[i], 
                         conSet['euler_error'], 
                         conSet['feedback_lin'], 
                         conSet['simple'])
    x = integrateRK4(x, u, dt, dynamics)
    
fig, ax = plt.subplots()
ax.plot(t, np.degrees(phi), 'r-', label='phi')
ax.plot(t, np.degrees(tht), 'g-', label='tht')
ax.plot(t, np.degrees(psi), 'b-', label='psi')
ax.plot(t, np.degrees(phi_c), 'r--', label='phi_c')
ax.plot(t, np.degrees(tht_c), 'g--', label='tht_c')
ax.plot(t, np.degrees(psi_c), 'b--', label='psi_c')
ax.grid()
ax.legend()
ax.set_ylabel('deg')
ax.set_xlabel('time (s)')
# fig.suptitle('%s Controller with %s Command' % (conType, comType))

# plt.show()
plt.savefig("%s-Controller-with-%s-Command.svg" % (conType, comType))