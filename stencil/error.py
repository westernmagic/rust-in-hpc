#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_field')
    parser.add_argument('out_field')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    baseline_path = here
    in_field_base = np.load(baseline_path + '/in_field_base.npy')
    out_field_base = np.load(baseline_path + '/out_field_base.npy')

    # path = here
    in_field = np.load(args.in_field)
    out_field = np.load(args.out_field)

    fig = plt.figure()
    plt.imshow((out_field_base - out_field)[:, :, 32], origin='lower')
    plt.colorbar()
    plt.show()
    # fig.savefig("diff.png")

    fig = plt.figure()
    plt.imshow(out_field[:, :, 32], origin='lower')
    plt.colorbar()
    plt.show()

    assert np.all(in_field == in_field_base)
    # assert np.all(np.abs(out_field - out_field_base) < 1e-4)
    print(np.amax(np.abs(out_field - out_field_base)))

if __name__ == '__main__':
    main()

# vim: set filetype=python expandtab tabstop=4 softtabstop=4 shiftwidth=4 :
