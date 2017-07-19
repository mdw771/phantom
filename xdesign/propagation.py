#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2016. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import warnings
import operator
from xdesign.util import gen_mesh
from xdesign.constants import PI
from numpy.fft import fft2, fftn, ifftn, fftshift, ifftshift

logger = logging.getLogger(__name__)

__author__ = "Ming Du"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['free_propagate',
           'far_propagate',
           'slice_modify',
           'slice_propagate',
           'get_kernel']


def slice_modify(simulator, delta_slice, beta_slice, wavefront):
    """Modify wavefront within a slice.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    delta_slice : ndarray
        Array of delta values.
    beta_slice : ndarray
        Array of beta values.
    wavefront : ndarray
        Complex wavefront.
    """
    delta_nm = simulator.voxel_nm[-1]
    kz = 2 * PI * delta_nm / simulator.lmbda_nm
    wavefront *= np.exp((kz * delta_slice) * 1j) * np.exp(-kz * beta_slice)
    return wavefront


def slice_propagate(simulator, wavefront, kernel=None):
    """"""
    delta_nm = simulator.voxel_nm[-1]
    wavefront = free_propagate(simulator, wavefront, delta_nm * 1e-7, kernel=kernel)
    return wavefront


def free_propagate(simulator, wavefront, dist, algorithm=None, kernel=None):
    """Free space propagation using convolutional algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    wavefront : ndarray
        The wavefront array.
    dist : float
        Propagation distance in cm.
    algorithm : str
        Force using the specified algorithm.
        Use 'TF' for transfer function, 'IR' for impulse response. 
    """
    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    if algorithm is None:
        l = np.prod(simulator.size_nm)**(1. / 3)
        crit_samp = lmbda_nm * dist_nm / l
        algorithm = 'TF' if simulator.mean_voxel_nm > crit_samp else 'IR'
    if algorithm == 'TF':
        return propagate_tf(simulator, wavefront, dist, kernel=kernel)
    elif algorithm == 'IR':
        return propagate_ir(simulator, wavefront, dist, kernel=kernel)
    else:
        raise ValueError('Invalid algorithm.')


def get_kernel(simulator, wavefront, dist):
    """
    Get Fresnel propagation kernel. Automatically judge whether to return IR or TF kernel.
    
    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    wavefront : ndarray
        The wavefront array.
    dist : float
        Propagation distance in cm.
    """

    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    l = np.prod(simulator.size_nm)**(1. / 3)
    crit_samp = lmbda_nm * dist_nm / l
    if simulator.mean_voxel_nm > crit_samp:
        return get_kernel_tf(simulator, wavefront, dist)
    else:
        return get_kernel_tf(simulator, wavefront, dist)


def get_kernel_tf(simulator, wavefront, dist):

    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    u_max = 1. / (2. * simulator.voxel_nm[0])
    v_max = 1. / (2. * simulator.voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], simulator.grid_delta.shape[1:3])
    H = np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm**2 * (u**2 - v**2)))

    return H


def propagate_tf(simulator, wavefront, dist, kernel=None):

    """Free space propagation using the transfer function algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    wavefront : ndarray
        The wavefront array.
    dist : float
        Propagation distance in cm.
    kernel : np.ndarray
        Fresnel propagator.
    """
    if kernel is not None:
        H = kernel
    else:
        H = get_kernel_tf(simulator, wavefront, dist)
    wavefront = ifftn(ifftshift(fftshift(fftn(wavefront)) * H))

    return wavefront


def get_kernel_ir(simulator, wavefront, dist):

    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    xmin, ymin = np.array(simulator.size_nm)[:2] / -2.
    dx, dy = simulator.voxel_nm[0:2]
    x = np.arange(xmin, xmin + simulator.size_nm[0], dx)
    y = np.arange(ymin, ymin + simulator.size_nm[1], dy)
    x, y = np.meshgrid(x, y)
    h = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(1j * k / (2 * dist_nm) * (x**2 + y**2))
    H = fft2(fftshift(h)) * simulator.voxel_nm[0] * simulator.voxel_nm[1]

    return H


def propagate_ir(simulator, wavefront, dist, kernel=None):

    """Free space propagation using the impulse response algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    wavefront : ndarray
        The wavefront array.
    dist : float
        Propagation distance in cm.
    """
    if kernel is not None:
        H = kernel
    else:
        H = get_kernel_ir(simulator, wavefront, dist)
    wavefront = fft2(fftshift(wavefront))
    wavefront = ifftshift(ifftn(wavefront * H))

    return wavefront


def far_propagate(simulator, wavefront, dist, pad=None):
    """Free space propagation using product Fourier algorithm. Suitable for far
    field propagation.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    wavefront : ndarray
        The wavefront array.
    dist : float
        Propagation distance in cm.
    """
    dist_nm = dist * 1.e7
    warnings.warn('This function is still under construction.')
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm

    if pad is not None:
        wavefront = np.pad(wavefront, pad, mode='edge')
        size_nm = np.array(wavefront.shape[:2]) * simulator.voxel_nm[:2]
    else:
        size_nm = simulator.size_nm

    xmin, ymin = size_nm[:2] / -2.
    dx, dy = simulator.voxel_nm[0:2]
    x = np.arange(xmin, xmin + size_nm[0], dx)
    y = np.arange(ymin, ymin + size_nm[1], dy)
    x, y = np.meshgrid(x, y)
    print(x)

    umin, vmin = lmbda_nm * dist_nm / (-2. * (simulator.size_nm[:2] / np.array(wavefront.shape)))
    du, dv = lmbda_nm * dist_nm / simulator.size_nm[:2]
    u = np.arange(umin, -umin, du)
    v = np.arange(vmin, -vmin, dv)
    print(u.shape)
    u, v = np.meshgrid(u, v)
    print(u.shape, u)

    wavefront = wavefront * np.exp(1j * k / (2 * dist_nm) * (x**2 + y**2))
    # wavefront = np.pad(wavefront, pad_width=512, mode='constant', constant_values=0)
    wavefront = fftshift(fft2(fftshift(wavefront)))
    # wavefront = wavefront[512:1024, 512:1024]
    wavefront = 1 / (1j * lmbda_nm * dist_nm) \
                * np.exp(1j * k / (2 * dist_nm) * (u**2 + v**2)) \
                * wavefront \
                * dx * dy

    return wavefront


def _far_propagate_2(grid, wavefront, lmd, z_um):
    """Free space propagation using product Fourier algorithm.
    """
    raise warnings.warn('DeprecatedWarning')

    N = grid.size_nm[1]
    M = grid.size_nm[2]
    D = N * grid.voxel_nm_y
    H = M * grid.voxel_nm_x
    f1 = wavefront

    V = N/D
    U = M/H
    d = np.arange(-(N-1)/2,(N-1)/2+1,1)*D/N
    h = np.arange(-(M-1)/2,(M-1)/2+1,1)*H/M
    v = np.arange(-(N-1)/2,(N-1)/2+1,1)*V/N
    u = np.arange(-(M-1)/2,(M-1)/2+1,1)*U/M

    f2 = np.fft.fftshift(np.fft.fft2(f1*np.exp(-1j*2*PI/lmd*np.sqrt(z_um**2+d**2+h[:,np.newaxis]**2))))*np.exp(-1j*2*PI*z_um/lmd*np.sqrt(1.+lmd**2*(v**2+u[:,np.newaxis]**2)))/U/V/(lmd*z_um)*(-np.sqrt(1j))
    d2,h2=v*lmd*z_um,u*lmd*z_um
    return f2
