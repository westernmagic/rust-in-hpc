from python2rust import zaxpy

import numpy as np

def main():
    a = 1.0 + 0.0j
    x = np.asarray([1.1 + 2.2j, 3.3 +  4.4j,   5.5  +  6.6j])
    y = np.asarray([7.7 + 8.8j, 9.9 + 10.10j, 11.11 + 12.12j])

    zaxpy(a, x, y)
    print(y)

if __name__ == '__main__':
    main()
