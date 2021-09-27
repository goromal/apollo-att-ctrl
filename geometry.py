import numpy as np

## Convert quaternion to rotation matrix.
#  @param q \f$4\times 1\f$ numpy array \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
#  @return \f$3\times 3\f$ numpy array.
def q2R(q):
    qw, qx, qy, qz = q[0,0], q[1,0], q[2,0], q[3,0]
    qxx = qx*qx
    qxy = qx*qy
    qxz = qx*qz
    qyy = qy*qy
    qyz = qy*qz
    qzz = qz*qz
    qwx = qw*qx
    qwy = qw*qy
    qwz = qw*qz
    return np.array([[1-2*qyy-2*qzz, 2*qxy-2*qwz, 2*qxz+2*qwy],
                     [2*qxy+2*qwz, 1-2*qxx-2*qzz, 2*qyz-2*qwx],
                     [2*qxz-2*qwy, 2*qyz+2*qwx, 1-2*qxx-2*qyy]])

## Convert rotation matrix to quaternion.
#  @param R \f$3\times 3\f$ numpy array.
#  @return \f$4\times 1\f$ numpy array \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
def R2q(R):
    R11, R12, R13 = R[0,0], R[0,1], R[0,2]
    R21, R22, R23 = R[1,0], R[1,1], R[1,2]
    R31, R32, R33 = R[2,0], R[2,1], R[2,2]

    if R11 + R22 + R33 > 0.0:
        s = 2 * np.sqrt(1 + R11 + R22 + R33)
        qw = s/4
        qx = 1/s*(R32-R23)
        qy = 1/s*(R13-R31)
        qz = 1/s*(R21-R12)
    elif R11 > R22 and R11 > R33:
        s = 2 * np.sqrt(1 + R11 - R22 - R33)
        qw = 1/s*(R32-R23)
        qx = s/4
        qy = 1/s*(R21+R12)
        qz = 1/s*(R31+R13)
    elif R22 > R33:
        s = 2 * np.sqrt(1 + R22 - R11 - R33)
        qw = 1/s*(R13-R31)
        qx = 1/s*(R21+R12)
        qy = s/4
        qz = 1/s*(R32+R23)
    else:
        s = 2 * np.sqrt(1 + R33 - R11 - R22)
        qw = 1/s*(R21-R12)
        qx = 1/s*(R31+R13)
        qy = 1/s*(R32+R23)
        qz = s/4
        
    q = np.array([[qw, qx, qy, qz]]).T

    if q[0,0] < 0:
        q *= -1

    return q 

def qL(q):
    qw, qx, qy, qz = q[0,0], q[1,0], q[2,0], q[3,0]
    return np.array([[qw, -qx, -qy, -qz],
                     [qx, qw, -qz, qy],
                     [qy, qz, qw, -qx],
                     [qz, -qy, qx, qw]])

class SO3(object):
    @staticmethod
    def random():
        return SO3(np.random.random((4,1)))

    @staticmethod
    def identity():
        return SO3(qw=1.0, qx=0, qy=0, qz=0)
        
    @staticmethod
    def fromAxisAngle(axis, angle):
        if not isinstance(axis, np.ndarray) or axis.shape != (3,1):
            raise TypeError('SO(3) axis-angle constructor requires the axis to be a 3x1 numpy array.')
        th2 = angle / 2.0
        arr = np.vstack((np.array([[np.cos(th2)]]), np.sin(th2)*axis))
        return SO3(arr)

    @staticmethod
    def fromEuler(roll, pitch, yaw):
        q_roll  = SO3.fromAxisAngle(np.array([[1],[0],[0]]), roll)
        q_pitch = SO3.fromAxisAngle(np.array([[0],[1],[0]]), pitch)
        q_yaw   = SO3.fromAxisAngle(np.array([[0],[0],[1]]), yaw)
        return q_yaw * q_pitch * q_roll

    @staticmethod
    def fromRotationMatrix(R):
        return SO3(R2q(R))

    @staticmethod
    def fromQuaternion(q):
        return SO3(q)

    @staticmethod
    def fromTwoUnitVectors(u, v):
        d = np.dot(u.T, v)
        if d < 0.99999999 and d > -0.99999999:
            invs = 1.0 / sqrt((2.0*(1.0+d)))
            xyz = np.cross(u, v*invs, axis=0)
            return SO3(qw=0.5/invs, qx=xyz[0,0], qy=xyz[1,0], qz=xyz[2,0])
        elif d < -0.99999999:
            return SO3(qw=0., qx=0., qy=1., qz=0.)
        else:
            return SO3(qw=1., qx=0., qy=0., qz=0.)

    def __init__(self, arr=None, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
        if arr is None:
            self.arr = np.array([[qw, qx, qy, qz]]).T
        else:
            if not isinstance(arr, np.ndarray) or arr.shape != (4,1):
                raise TypeError('SO(3) default constructor requires a 4x1 numpy array quaternion [qw qx qy qz]^T.')
            self.arr = arr 
        self.arr /= np.linalg.norm(self.arr)
        if self.arr[0,0] < 0:
            self.arr *= -1
            
    def __repr__(self):
        return "SO(3): qw=%f, qx=%f, qy=%f, qz=%f" % tuple(self.arr.T[0])

    def R(self):
        return q2R(self.arr)

    def q(self):
        return self.arr

    def inverse(self):
        return SO3(qw=self.arr[0,0], qx=-self.arr[1,0], qy=-self.arr[2,0], qz=-self.arr[3,0])

    def invert(self):
        self.arr[1:,:] *= -1.0

    ## Convert SO3 to Euler representation (roll, pitch, yaw).
    #
    #  Uses the flight dynamics convention: roll about x -> pitch about y -> yaw about z (http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles)
    def toEuler(self):
        qw = self.arr[0,0]
        qx = self.arr[1,0]
        qy = self.arr[2,0]
        qz = self.arr[3,0]

        yr = 2.*(qw*qx + qy*qz)
        xr = 1 - 2*(qx**2 + qy**2)
        roll = np.arctan2(yr, xr)

        sp = 2.*(qw*qy - qz*qx)
        pitch = np.arcsin(sp)

        yy = 2.*(qw*qz + qx*qy)
        xy = 1 - 2.*(qy**2 + qz**2)
        yaw = np.arctan2(yy, xy)

        return roll, pitch, yaw

    def __mul__(self, other):
        if not isinstance(other, SO3):
            raise TypeError('SO(3) multiplication is only valid with another SO(3) object.')
        return SO3(np.dot(qL(self.arr), other.arr))

    def __add__(self, other):
        if isinstance(other, np.ndarray) and other.shape == (3,1):
            return self * SO3.Exp(other)
        else:
            raise TypeError('SO(3) can only be perturbed by a 3x1 numpy array tangent space vector.')

    def __sub__(self, other):
        if isinstance(other, SO3):
            return SO3.Log(other.inverse() * self)
        else:
            raise TypeError('SO(3) can only be differenced by another SO(3) object.')

    @staticmethod
    def hat(o):
        if isinstance(o, np.ndarray) and o.shape == (3,1):
            return np.array([[0, -o[2,0], o[1,0]],
                             [o[2,0], 0, -o[0,0]],
                             [-o[1,0], o[0,0], 0]])
        else:
            raise TypeError('The SO(3) hat operator must take in a 3x1 numpy array tangent space vector.')

    @staticmethod
    def vee(Omega):
        if isinstance(Omega, np.ndarray) and Omega.shape == (3,3) and Omega.T == -Omega:
            return np.array([[Omega[2,1]],[Omega[0,2]],[Omega[1,0]]])
        else:
            raise TypeError('The SO(3) vee operator must take in a member of the so(3) Lie algebra.')

    @staticmethod
    def log(X):
        return SO3.hat(SO3.Log(X))

    @staticmethod
    def Log(X):
        qv = X.arr[1:,:]
        qw = X.arr[0,0]
        n = np.linalg.norm(qv)
        if n > 0.0:
            return qv * 2.0 * np.arctan2(n, qw) / n
        else:
            return np.zeros((3,1))

    @staticmethod
    def exp(v):
        return SO3.Exp(SO3.vee(v))

    @staticmethod
    def Exp(v):
        th = np.linalg.norm(v)
        q = np.zeros((4,1))
        if th > 0.0:
            u = v / th
            q[0] = np.cos(th/2)
            q[1:,:] = u * np.sin(th/2)
        else:
            q[0] = 1.0
        return SO3(q)
        
class SE3(object):
    @staticmethod
    def random():
        return SE3(np.random.random((7,1)))

    @staticmethod
    def identity():
        return SE3(tx=0, ty=0, tz=0, qw=1.0, qx=0, qy=0, qz=0)

    @staticmethod
    def fromTransformMatrix(T):
        return SE3(np.vstack((T[0:3,3:], R2q(T[0:3,0:3]))))

    @staticmethod
    def fromTranslationAndRotation(*args):
        if len(args) == 1:
            return SE3(args[0])
        elif len(args) == 2 and isinstance(args[1], np.ndarray):
            return SE3(np.vstack((args[0], args[1])))
        elif len(args) == 2 and isinstance(args[1], SO3):
            return SE3(np.vstack((args[0], args[1].q())))
        else:
            raise TypeError('Translation/Quaternion allowed types are (7x1 array), (3x1 array, 4x1 array), (3x1 array, SO3).')

    def __init__(self, arr=None, tx=0, ty=0, tz=0, qw=1.0, qx=0, qy=0, qz=0):
        if arr is None:
            self.t = np.array([[tx, ty, tz]]).T
            self.q = SO3(qw=qw, qx=qx, qy=qy, qz=qz)
        else:
            if not isinstance(arr, np.ndarray) or arr.shape != (7,1):
                raise TypeError('SE(3) default constructor requires a 7x1 numpy array [tx ty tz qw qx qy qz]^T.')
            self.t = arr[0:3,:]
            self.q = SO3(arr[3:7,:])
            
    def __repr__(self):
    	tqtup = tuple(self.t) + tuple(self.q.arr)
    	return "SE(3): tx=%f, ty=%f, tz=%f, qw=%f, qx=%f, qy=%f, qz=%f" % tqtup

    def T(self):
        return np.vstack((np.hstack((self.q.R(), self.t)), np.array([[0,0,0,1.0]])))

    def tq(self):
        return np.vstack((self.t, self.q.q()))

    def transform(self, v):
        return np.dot(self.q.R(), v) + self.t

    def inverse(self):
        return SE3.fromTranslationAndRotation(-np.dot(self.q.inverse().R(), self.t), self.q.inverse())

    def invert(self):
        self.q.invert()
        self.t = -np.dot(self.q.R(), self.t)
 
    def __mul__(self, other):
        if not isinstance(other, SE3):
            raise TypeError('SE(3) multiplication is only valid with another SE(3) object.')
        return SE3.fromTranslationAndRotation(self.t + np.dot(self.q.R(), other.t), self.q * other.q)

    ## \f$\oplus\f$
    #
    # More details.
    def __add__(self, other):
        if isinstance(other, np.ndarray) and other.shape == (6,1):
            return self * SE3.Exp(other)
        else:
            raise TypeError('SE(3) can only be perturbed by a 6x1 numpy array tangent space vector.')

    def __sub__(self, other):
        if isinstance(other, SE3):
            return SE3.Log(other.inverse() * self)
        else:
            raise TypeError('SE(3) can only be differenced by another SE(3) object.')

    @staticmethod
    def hat(o):
        if isinstance(o, np.ndarray) and o.shape == (6,1):
            return np.vstack((np.hstack((SO3.hat(o[3:,:]),o[:3,:])),np.zeros((1,4))))
        else:
            raise TypeError('The SE(3) hat operator must take in a 6x1 numpy array tangent space vector.')

    @staticmethod
    def vee(Omega):
        if isinstance(Omega, np.ndarray) and Omega.shape == (4,4):
            return np.vstack((Omega[:3,3:], SO3.vee(Omega[:3,:3])))
        else:
            raise TypeError('The SE(3) vee operator must take in a member of the se(3) Lie algebra.')

    @staticmethod
    def log(X):
        return SE3.hat(SE3.Log(X))

    @staticmethod
    def Log(X):
        w = SO3.Log(X.q)
        th = np.linalg.norm(w)
        W = SO3.hat(w)

        if th > 0:
            a = np.sin(th)/th
            b = (1-np.cos(th))/(th**2)
            c = (1-a)/(th**2)
            e = (b-2*c)/(2*a)
            Jl_inv = np.eye(3) - 0.5 * W + e * np.dot(W, W)
        else:
            Jl_inv = np.eye(3)

        return np.vstack((np.dot(Jl_inv, X.t), w))

    @staticmethod
    def exp(v):
        return SE3.Exp(SE3.vee(v))

    @staticmethod
    def Exp(v):
        rho = v[:3,:]
        w = v[3:,:]
        q = SO3.Exp(w)
        W = SO3.hat(w)
        th = np.linalg.norm(w)

        if th > 0:
            a = np.sin(th)/th
            b = (1-np.cos(th))/(th**2)
            c = (1-a)/(th**2)
            Jl = a * np.eye(3) + b * W + c * np.dot(w, w.T)
        else:
            Jl = np.eye(3)

        return SE3.fromTranslationAndRotation(np.dot(Jl, rho), q)
        
def test_SO3():
    R1 = SO3.random()
    print(R1)
    R2 = R1 + np.array([[0.5, 0.2, 0.1]]).T
    print(R2)
    print(R2-R1)
    
def test2_SO3():
    q = SO3.fromEuler(-1.2, 0.6, -0.4)
    print(q.toEuler())
    
def test_SE3():
    np.random.seed(144440)
    Trand = SE3.random()
    Trond = SE3.random()
    dT = Trond-Trand
    T2 = Trand + dT

    print(Trand * Trond)
    print()
    print(Trand.inverse())
    print()
    print(dT)
    print()

    print(Trond)
    print()
    print(SE3.Exp(SE3.Log(Trond)))
    print()
    print(T2)
    print()
    print(dT)
    print()
    print(SE3.Log(SE3.Exp(dT)))
    
if __name__ == '__main__':
	# test_SO3()
	# print()
	# test_SE3()
	test2_SO3()