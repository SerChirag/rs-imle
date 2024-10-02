import numpy as np
from scipy import interpolate

class Angle_Generator:
    def __init__(self, dim):
        self.dim = dim
        curve, inverse_cdf = self.sample(self.f)

        self.curve = curve
        self.inverse_cdf = inverse_cdf

    def f(self, x):
        # does not need to be normalized
        return np.sin(x)** (self.dim - 2)

    def sample(self, g):
        x = np.linspace(0,np.pi,10000)
        y = g(x)                        # probability density function, pdf
        cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
        cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
        inverse_cdf = interpolate.interp1d(cdf_y,x) # this is a function
        curve = interpolate.interp1d(x,cdf_y) 
        self.curve = curve
        self.inverse_cdf = inverse_cdf
        return curve, inverse_cdf

    def return_samples(self, N=1000, angle_low = 0, angle_high = np.pi):
        # let's generate some samples according to the chosen pdf, f(x)
        u_low = self.curve(angle_low)
        u_high = self.curve(angle_high)
        uniform_samples = np.random.uniform(u_low, u_high,int(N))
        required_samples = self.inverse_cdf(uniform_samples)
        required_samples = required_samples.astype(np.float32)
        return required_samples
