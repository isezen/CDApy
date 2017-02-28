
import numpy as np # import numpy library

# Answer 1
# create normally distributed random numbers
x = np.random.normal(2, 3, 100) # mean, sd, count
n = len(x)

mean = np.sum(x)/n # method 1
mean2 = sum(x.tolist())/n # method 2

var = np.sum(((x - mean)**2)/n) # Variance
sd = np.sqrt(var)

# compare results
np.mean(x) == mean # is this always TRUE?
np.std(x) == sd # is this always TRUE?
np.var(x) == var # is this always TRUE?

# -----------------------------------------------------------------------------
# Answer 2
import matplotlib.pyplot as plt # for plotting
import matplotlib.mlab as mlab

x = np.arange(2, 5000, 3)
x = np.remainder(x, 13)
mu = np.median(x)
sigma = np.var(x)

n, bins, patches = plt.hist(x, facecolor='green', alpha=0.75)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.title(r'$\mathrm{Histogram\ of\ x:}\ \mu=6,\ \sigma=14$')
plt.grid(True)

# -----------------------------------------------------------------------------
# Answer #3
from scipy import stats

x = np.arange(4, 250)
h = (np.log(x) - 1)/np.sqrt(x)
g = np.exp(np.sqrt(h))
f = np.sin(g) ** np.cos(g)

sigma = np.var(f)
sd = np.std(f)
stats.describe(f) # equivalent of summary function in R

# plot histogram
plt.hist(f, facecolor = 'red', alpha = 0.75)
plt.axvline(x=np.median(f))
plt.show()

plt.plot(x, f) # plot f versus x.
plt.xlabel('x-values')
plt.ylabel('f(x)')
plt.show()

# -----------------------------------------------------------------------------
# Answer #4

y = [87, 23]
x = [155, 2543]
x1 = 240
# what is y1?

# formula of a line between 2 points
# y = mx + b

# slope of the line
m = (y[1] - y[0])/float(x[1] - x[0]) # why we did floating division here?
# intercept
b = y[0] - m * x[0]
# or
b = (x[1] * y[0] - x[0] * y[1])/float(x[1] - x[0])
y1 = m * x1 + b
print y1


# by scipy -> linear interpolation
from scipy import interpolate
f = interpolate.interp1d(x, y)
callable(f) # if f is a function, returns True
print f(x1) # use f function to calculate y value of x1

# -----------------------------------------------------------------------------
# Answer #5

import string as s
letters = list(s.ascii_lowercase)

shift = 2
crypted_text = list('yknnmqoogp')

# crypt
index = range((shift), len(letters)) + range(0, shift)
chipper = [letters[i] for i in index]
d = dict(zip(letters, chipper))
crypted_text = [d[i] for i in list('willkommen')]
print ''.join(crypted_text)

# decrypt
index = range((shift), len(letters)) + range(0, shift)
chipper = [letters[i] for i in index]
d = dict(zip(chipper, letters))
decrypted_text = [d[i] for i in crypted_text]
print ''.join(decrypted_text)

# -----------------------------------------------------------------------------
# Answer #6
import math as m

a = 0; b = 2; n = 1e8
seq = np.linspace(a, b, n + 1)
intg = ((b - a)/float(n)) * np.sum(np.sin(seq))
dif = intg - (m.cos(a) - m.cos(b))

# OR
from scipy.integrate import quad
quad(m.sin, 0, 2) # second term is absolute error

