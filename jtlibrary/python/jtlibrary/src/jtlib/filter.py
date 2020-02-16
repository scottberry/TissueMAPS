# Copyright (C) 2016 University of Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import numpy as np


def _g(y, x, sigma):
    return np.e**(  -( y**2 + x**2 ) / ( 2 * sigma**2 ) )


def gaussian_2d(size, sigma):
    '''Creates a 2D Gaussian convolution filter.

    Parameters
    ----------
    size: int
        width and height of the filter kernel
    sigma: float
        standard deviation

    Returns
    -------
    numpy.ndarray[numpy.float]
        convolution kernel

    Note
    ----
    Implements the Matlab ``fspecial`` function with ``"gaussian"`` argument.
    '''
    k = np.zeros((size, size), np.float)
    r = range(-int(np.floor(float(size)/2)), int(np.ceil(float(size)/2)))
    for i, y in enumerate(r):
        for j, x in enumerate(r):
            k[i, j] = _g(y, x, sigma)
    return k / np.sum(k)


def log_2d(size, sigma):
    '''Creates a 2D Laplacian of Gaussian convolution filter.

    Parameters
    ----------
    size: int
        width and height of the filter kernel
    sigma: float
        standard deviation of the Gaussian component

    Returns
    -------
    numpy.ndarray[numpy.float]
        convolution kernel

    Note
    ----
    Implements the Matlab ``fspecial`` function with ``"log"`` argument.
    '''
    k = np.zeros((size, size))
    r = range(-int(np.floor(float(size)/2)), int(np.ceil(float(size)/2)))
    g = np.sum([_g(x, y, sigma) for x, y in itertools.product(r, r)])
    for i, y in enumerate(r):
        for j, x in enumerate(r):
            k[i, j] = (
                ( _g(y, x, sigma) / g ) * ( y**2 + x**2 - 2 * sigma**2 ) /
                ( sigma**4 )
            )
    return k - np.sum(k) / k.size


def _g3d(x, y, z, sigma):
    return np.e**(  -( ( x**2 / (2 * sigma[0]**2) ) +
                       ( y**2 / (2 * sigma[1]**2) ) +
                       ( z**2 / (2 * sigma[2]**2) ) ) )


def log_3d(size, sigma):
    '''Creates a 3D Laplacian of Gaussian convolution filter.

    Parameters
    ----------
    size: int
        width and height of the filter kernel
    sigma: float
        standard deviation of the Gaussian component

    Returns
    -------
    numpy.ndarray[numpy.float]
        convolution kernel

    '''
    f = np.zeros((size, size, size))
    r = range(-int(np.floor(float(size)/2)), int(np.ceil(float(size)/2)))
    g = np.sum([_g3d(x, y, z, sigma) for x, y, z in itertools.product(r, r, r)])
    for i, x in enumerate(r):
        for j, y in enumerate(r):
            for k, z in enumerate(r):
                f[i,j,k] = ( ( _g3d(x, y, z, sigma) / g ) * (
                    ( x**2 / sigma[0]**4 ) - ( 1 / sigma[0]**2 ) +
                    ( y**2 / sigma[1]**4 ) - ( 1 / sigma[1]**2 ) +
                    ( z**2 / sigma[2]**4 ) - ( 1 / sigma[2]**2 )
                ) )
    return f - np.sum(f) / f.size


