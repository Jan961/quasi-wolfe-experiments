import numpy as np
from scipy.interpolate import PchipInterpolator


def generate_lipschitz_estimate(dimension, override_assertion=False):

    if not override_assertion:
        assert dimension >= 127, "Dimension should be at least 127"

    slope = 0.8142014074788393

    intercept = 0.19404509824465777
    dimensions = np.array([  127. ,  206. ,  335.  , 545. ,  885. , 1438.,  2335.,  3792. , 6158.,10000.])

    medians = np.array([9.28521402 ,11.764596 ,  14.96667777 ,19.06256585 ,24.28337195,
     30.94219835 ,39.43544396 ,50.255145  , 64.04513481 ,81.61418585])

    spline = PchipInterpolator(slope * np.sqrt(dimensions) + intercept, medians)

    return spline(slope * np.sqrt(dimension)+ intercept)
