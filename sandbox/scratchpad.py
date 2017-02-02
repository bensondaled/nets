from soup import *

##
def cross_entropy(y,a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))
def quadratic(y,a):
    return (y-a)**2
def linear(y,a):
    return np.abs(y-a)
##

for y in [0.3]:#np.arange(0,1.1,0.1):
    a = np.arange(0,1,0.0001) # empirical values

    ce = cross_entropy(y=y,a=a)
    qu = quadratic(y=y,a=a)
    li = linear(y=y,a=a)

    pl.plot(a, ce)
    pl.plot(a, qu)
    pl.plot(a, li)

##
