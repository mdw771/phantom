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
from scipy.interpolate import interp2d
# from numpy.fft import fft2, fftn, ifftn, fftshift, ifftshift
from pyfftw.interfaces.numpy_fft import fftshift, ifftshift, fftn, ifftn, fft, ifft, fft2, ifft2

logger = logging.getLogger(__name__)

__author__ = "Ming Du"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['free_propagate',
           'far_propagate',
           'slice_modify',
           'slice_propagate',
           'get_kernel',
           'propagate_tf',
           'propagate_ir']


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
        print(algorithm)
    if algorithm == 'TF':
        return propagate_tf(simulator, wavefront, dist, kernel=kernel)
    elif algorithm == 'IR':
        return propagate_ir(simulator, wavefront, dist, kernel=kernel)
    else:
        raise ValueError('Invalid algorithm.')


def get_kernel(simulator, dist):
    """Get Fresnel propagation kernel. Automatically judge whether to return IR or TF kernel.
    
    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """

    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    l = np.prod(simulator.size_nm)**(1. / 3)
    crit_samp = lmbda_nm * dist_nm / l
    if simulator.mean_voxel_nm > crit_samp:
        return get_kernel_tf(simulator, dist)
    else:
        return get_kernel_ir(simulator, dist)


def get_kernel_tf_real(simulator, dist):

    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    xmin, ymin = np.array(simulator.size_nm)[:2] / -2.
    dx, dy = simulator.voxel_nm[0:2]
    x = np.arange(xmin, xmin + simulator.size_nm[0], dx)
    y = np.arange(ymin, ymin + simulator.size_nm[1], dy)
    x, y = np.meshgrid(x, y)
    h = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))

    return h


def get_kernel_tf(simulator, dist):
    """Get Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    u_max = 1. / (2. * simulator.voxel_nm[0])
    v_max = 1. / (2. * simulator.voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], simulator.grid_delta.shape[0:2])
    H = np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm**2 * (u**2 + v**2)))

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
        H = get_kernel_tf(simulator, dist)
    wavefront = ifftn(ifftshift(fftshift(fftn(wavefront)) * H))

    return wavefront


def get_kernel_ir(simulator, dist):

    """
    Get Fresnel propagation kernel for IR algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    dist_nm = dist * 1e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    ymin, xmin = np.array(simulator.size_nm)[:2] / -2.
    dy, dx = simulator.voxel_nm[0:2]
    x = np.arange(xmin, xmin + simulator.size_nm[1], dx)
    y = np.arange(ymin, ymin + simulator.size_nm[2], dy)
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
        H = get_kernel_ir(simulator, dist)
    wavefront = fft2(fftshift(wavefront))
    wavefront = ifftshift(ifftn(wavefront * H))

    return wavefront


def far_propagate(simulator, wavefront, dist, pad=None, return_coords=False):
    """
    Modified single Fourier transform propagation suitable for all range
    of numerical aperture. Only proper for far field propagation. 
    Reference: 
    S. Ruschin and Y. M. Engelberg, J. Opt. Soc. Am. A, JOSAA, vol. 21, 
    no. 11, pp. 2135–2145, Nov. 2004.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    wavefront : ndarray
        The wavefront array.
    dist : float
        Propagation distance in cm.
    return_coords : bool
        Whether to return the coordinates of observation plane. 
    """
    dist_nm = dist * 1.e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    wavedc = wavefront[0, 0]
    wavefront = wavefront - wavefront[0, 0]

    original_shape = wavefront.shape
    if pad is not None:
        wavefront = np.pad(wavefront, pad, mode='constant', constant_values=0)
        size_nm = np.array(wavefront.shape[:2]) * simulator.voxel_nm[:2]
    else:
        size_nm = simulator.size_nm

    xmin, ymin = size_nm[:2] / -2.
    dx, dy = simulator.voxel_nm[0:2]
    x = np.arange(xmin, xmin + size_nm[0], dx)
    y = np.arange(ymin, ymin + size_nm[1], dy)
    x, y = np.meshgrid(x, y)

    u_max = 1. / (2. * simulator.voxel_nm[0])
    v_max = 1. / (2. * simulator.voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], simulator.grid_delta.shape[0:2])

    corr_factor = np.sqrt(1 - lmbda_nm**2 * (u**2 + v**2))
    x2 = lmbda_nm * u * dist_nm / corr_factor
    y2 = lmbda_nm * v * dist_nm / corr_factor

    r2 = np.sqrt(dist_nm**2 + x2**2 + y2**2)

    wavefront = wavefront * np.exp(1j * k / (2 * dist_nm) * (x**2 + y**2))
    wavefront = ifftshift(fft2(fftshift(wavefront)))
    wavefront = wavefront * dist_nm / (1j * lmbda_nm * r2**2)
    wavefront = wavefront * np.exp(1j * k * r2)

    wavedc *= np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm ** 2 * (u_max**2 - v_max**2)))

    wavefront += wavedc

    if pad is not None:
        wavefront = wavefront[pad:pad+original_shape[0], pad:pad+original_shape[1]]

    # interpolation back to linear grid

    f = interp2d(x2[0, :], y2[:, 0], wavefront)
    umin, vmin = lmbda_nm * dist_nm / (-2. * (simulator.voxel_nm[:2]))
    du, dv = lmbda_nm * dist_nm / simulator.size_nm[:2]
    x2 = np.arange(umin, -umin, du)
    y2 = np.arange(vmin, -vmin, dv)
    wavefront = f(x2, y2)

    if return_coords:
        return wavefront, (x2, y2)
    else:
        return wavefront


def far_propagate_low_na(simulator, wavefront, dist, pad=None, return_coords=False):
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
    return_coords : bool
        Whether to return the coordinates of observation plane. 
    """
    dist_nm = dist * 1.e7
    lmbda_nm = simulator.lmbda_nm
    k = 2 * PI / lmbda_nm
    wavedc = wavefront[0, 0]
    wavefront = wavefront - wavefront[0, 0]

    original_shape = wavefront.shape
    if pad is not None:
        wavefront = np.pad(wavefront, pad, mode='constant', constant_values=0)
        size_nm = np.array(wavefront.shape[:2]) * simulator.voxel_nm[:2]
    else:
        size_nm = simulator.size_nm

    xmin, ymin = size_nm[:2] / -2.
    dx, dy = simulator.voxel_nm[0:2]
    x = np.arange(xmin, xmin + size_nm[0], dx)
    y = np.arange(ymin, ymin + size_nm[1], dy)
    x, y = np.meshgrid(x, y)

    umin, vmin = lmbda_nm * dist_nm / (-2. * (simulator.size_nm[:2] / np.array(wavefront.shape)))
    du, dv = lmbda_nm * dist_nm / simulator.size_nm[:2]
    u = np.arange(umin, -umin, du)
    v = np.arange(vmin, -vmin, dv)
    u, v = np.meshgrid(u, v)

    wavefront = wavefront * np.exp(1j * k / (2 * dist_nm) * (x**2 + y**2))
    wavefront = ifftshift(fft2(fftshift(wavefront)))
    wavefront = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) \
                * np.exp(1j * k / (2 * dist_nm) * (u**2 + v**2)) \
                * wavefront \
                * dx * dy

    u_max = 1. / (2. * simulator.voxel_nm[0])
    v_max = 1. / (2. * simulator.voxel_nm[1])
    wavedc *= np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm ** 2 * (u_max**2 - v_max**2)))

    wavefront += wavedc

    if pad is not None:
        wavefront = wavefront[pad:pad+original_shape[0], pad:pad+original_shape[1]]

    if return_coords:
        return wavefront, (u, v)
    else:
        return wavefront

