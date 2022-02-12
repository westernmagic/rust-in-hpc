#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset=(3 + rank) * 32 // nbits
    data = np.fromfile(filename, dtype=np.float32 if nbits == 32 else np.float64, \
                       count=nz * ny * nx + offset)
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    baseline_path = here
    in_field_base = read_field_from_file(baseline_path + '/in_field_base.dat')
    out_field_base = read_field_from_file(baseline_path + '/out_field_base.dat')

    path = here
    in_field = read_field_from_file(path + '/in_field.dat')
    out_field = read_field_from_file(path + '/out_field.dat')

    fig = plt.figure()
    plt.imshow((out_field_base - out_field)[32, :, :], origin='lower')
    plt.colorbar()
    plt.show()
    # fig.savefig("diff.png")

    fig = plt.figure()
    plt.imshow(out_field[32, :, :], origin='lower')
    plt.colorbar()
    plt.show()

    assert np.all(in_field == in_field_base)
    # assert np.all(np.abs(out_field - out_field_base) < 1e-4)
    print(np.amax(np.abs(out_field - out_field_base)))

if __name__ == '__main__':
    main()

# vim: set filetype=python expandtab tabstop=4 softtabstop=4 shiftwidth=4 :
