import numpy as np
from numbers import Number

# https://github.com/goromal/math-utils-lib/blob/master/include/math-utils-lib/modeling.h

class Differentiator(object):
    def __init__(self, shape=None, sigma=0.05):
        self.shape = shape
        self.sigma = sigma
        self.deriv_curr = None
        self.deriv_prev = None
        self.val_prev   = None
        self.initialized = False

    def calculate(self, val, Ts):
        if self.initialized:
            self.deriv_curr = (2 * self.sigma - Ts) / (2 * self.sigma + Ts) * self.deriv_prev + \
                              2 / (2 * self.sigma + Ts) * (val - self.val_prev)
        else:
            self.deriv_curr = np.zeros(self.shape) if not self.shape is None else 0
            self.initialized = True
        self.deriv_prev = self.deriv_curr
        self.val_prev = val
        return self.deriv_curr
    
def timeDerivative(t, y, sigma=0.05, tangent_dim=None):
    t_data = list()
    t_prev = 0
        
    if isinstance(t, np.ndarray):
        if len(t.shape) == 1:
            for i in range(t.size):
                t_data.append(t[i])
        elif len(t.shape) == 2 and t.shape[0] == 1:
            for i in range(t.size):
                t_data.append(t[0,i])
        else:
            raise TypeError('When t is a np.ndarray, it must be an array or a row vector.')
    elif isinstance(t, list):
        t_data = t
    else:
        raise TypeError('t must be a list, np array, or np row vector.')

    n = len(t_data)
    shape = None
    D = None
    ydot = None

    if isinstance(y, np.ndarray):
        ydot = np.empty_like(y)
        if len(y.shape) == 1 and y.size == n:
            D = Differentiator(shape=shape, sigma=sigma)
            for i in range(n):
                Ts = t_data[i] - t_prev
                ydot[i] = D.calculate(y[i], Ts)
                t_prev = t_data[i]
        elif len(y.shape) == 2 and y.shape[1] == n:
            shape = (y.shape[0], 1)
            D = Differentiator(shape=shape, sigma=sigma)
            for i in range(n):
                Ts = t_data[i] - t_prev
                ydot[:,i:i+1] = D.calculate(y[:,i:i+1], Ts)
                t_prev = t_data[i]
        else:
            raise TypeError('y must contain the same number of elements as t.')
    elif isinstance(y, list):
        if len(y) == n:
            if isinstance(y[0], Number):
                ydot = list()
                D = Differentiator(shape=shape, sigma=sigma)
                for i in range(n):
                    Ts = t_data[i] - t_prev
                    ydot.append(D.calculate(y[i], Ts))
                    t_prev = t_data[i]
            elif isinstance(y[0], np.ndarray):
                if len(y[0].shape) == 1 or (len(y[0].shape) == 2 and y[0].shape[1] == 1):
                    ydot = list()
                    shape = y[0].shape
                    D = Differentiator(shape=shape, sigma=sigma)
                    for i in range(n):
                        Ts = t_data[i] - t_prev
                        ydot.append(D.calculate(y[i], Ts))
                        t_prev = t_data[i]
                else:
                    raise TypeError('A list of arrays for y must consist of numpy arrays or column vectors.')
            else:
                if tangent_dim is None:
                    raise AttributeError('A list of non-numeric y requires tangent_dim to be specified.')
                else:
                    ydot = list()
                    shape = (tangent_dim, 1)
                    D = Differentiator(shape=shape, sigma=sigma)
                    for i in range(n):
                        Ts = t_data[i] - t_prev
                        ydot.append(D.calculate(y[i], Ts))
                        t_prev = t_data[i]
        else:
            raise TypeError('y must contain the same number of elements as t.')
    else:
        raise TypeError('y must be a list, np array, or np row vector.')
    
    return ydot                    
                
def integrateRK4(x, u, dt, f):
	k1 = f(x, u)
	k2 = f(x + k1*dt/2, u)
	k3 = f(x + k2*dt/2, u)
	k4 = f(x + k3*dt, u)
	return x + (k1 + 2*k2 + 2*k3 + k4)*dt/6

# https://github.com/goromal/matlab_utilities/blob/master/math/GeneralizedInterpolator.m

class InterpolatorBase(object):
    # list and list
    def __init__(self, t_data, y_data):
        self.t_data = t_data
        self.n = len(t_data)
        self.y_data = y_data
        self.i = 0

    def y(self, t):
        if t < self.t_data[0]:
            return self.y_data[0]
        if t > self.t_data[-1]:
            return self.y_data[-1]
        if t in self.t_data:
            return self.y_data[self.t_data.index(t)]
        for idx in range(self.n-1):
            self.i = idx
            ti = self.t_data[self.i]
            if ti < t and self._ti(self.i+1) > t:
                return self._interpy(t)
        return None

    def _interpy(self, t):
        return self._yi(self.i) + self._dy(t)

    def _dy(self, t):
        return None

    def _yi(self, i):
        if i < 0:
            return self.y_data[0]
        elif i >= self.n:
            return self.y_data[-1]
        else:
            return self.y_data[i]

    def _ti(self, i):
        if i < 0:
            return self.t_data[0] - 1.0
        elif i >= self.n:
            return self.t_data[-1] + 1.0
        else:
            return self.t_data[i]

class ZeroOrderInterpolator(InterpolatorBase):
    def __init__(self, t_data, y_data, zero_obj):
        super(ZeroOrderInterpolator, self).__init__(t_data, y_data)
        self.zero_obj = zero_obj

    def _dy(self, t):
        return self.zero_obj

class LinearInterpolator(InterpolatorBase):
    def __init__(self, t_data, y_data):
        super(LinearInterpolator, self).__init__(t_data, y_data)

    def _dy(self, t):
        t1 = self._ti(self.i)
        t2 = self._ti(self.i+1)
        y1 = self._yi(self.i)
        y2 = self._yi(self.i+1)
        return (t - t1) / (t2 - t1) * (y2 - y1)

class SplineInterpolator(InterpolatorBase):
    def __init__(self, t_data, y_data):
        super(SplineInterpolator, self).__init__(t_data, y_data)

    def _dy(self, t):
        t0 = self._ti(self.i-1)
        t1 = self._ti(self.i)
        t2 = self._ti(self.i+1)
        t3 = self._ti(self.i+2)
        y0 = self._yi(self.i-1)
        y1 = self._yi(self.i)
        y2 = self._yi(self.i+1)
        y3 = self._yi(self.i+2)
        return (t-t1)/(t2-t1)*((y2-y1) + \
             (t2-t)/(2*(t2-t1)**2)*(((t2-t)*(t2*(y1-y0)+t0*(y2-y1)-t1*(y2-y0)))/(t1-t0) + \
             ((t-t1)*(t3*(y2-y1)+t2*(y3-y1)-t1*(y3-y2)))/(t3-t2)))