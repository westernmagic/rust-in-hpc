from python2rust import zaxpy

import numpy as np

def main():
    a = 10.
    x = np.asarray([1., 2., 3.])
    y = np.asarray([4., 5., 6.])

    zaxpy(a, x, y)
    print(y)

if __name__ == '__main__':
    main()
