from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import dxchange
import time

from xdesign.material import XraylibMaterial, CustomMaterial
from xdesign.geometry import *
from xdesign.phantom import Phantom
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator


# materials
myelin = XraylibMaterial('C5H8O8P', 1.001)
protein = XraylibMaterial('H48.6C32.9N8.9O8.9S0.6', 1.35)
water = XraylibMaterial('H2O', 1)
nucleic_acid = XraylibMaterial('C4H5.25ON3.75', 1.7)

# model settings
n_neurons = 40
neuron_size_range = (10, 30)
grid_size = (512, 512, 512)
energy = 25 # kev

# build model
# get neuron list
grid_size = np.array(grid_size)

neuron_pos_ls = []
neuron_size_ls = []
for i in range(n_neurons):
    pos = np.random.rand(3) * grid_size
    size = np.random.rand() * neuron_size_range[1] + neuron_size_range[0]
    neuron_pos_ls.append(pos)
    neuron_size_ls.append(size)
neuron_pos_ls = np.array(neuron_pos_ls)

substrate = Cuboid_3d(x1=Point([0, 0, 0]),
                      x2=Point(list(grid_size)))
phantom = Phantom(geometry=substrate,
                  material=protein)

# find pairs of axon connections
connections = []
for i in range(n_neurons):
    # calculate distance to other neurons
    this_pos = neuron_pos_ls[i]
    dist_ls = np.sum((this_pos - neuron_pos_ls) ** 2, axis=1)
    dist_ls[i] = np.inf
    while True:
        if dist_ls.min() == np.inf:
            break
        else:
            ind = np.argmin(dist_ls)
            connect = {i, ind}
            if connect not in connections:
                connections.append(connect)
                break
            else:
                dist_ls[ind] = np.inf

print(connections)

# build axons
for connect in connections:
    connect = list(connect)
    axon = Rod_3d(x1=Point(neuron_pos_ls[connect[0]]),
                  x2=Point(neuron_pos_ls[connect[1]]),
                  radius=3)
    axon_pht = Phantom(geometry=axon,
                       material=myelin)
    phantom.children.append(axon_pht)

# build cells
for i in range(n_neurons):
    pos = neuron_pos_ls[i]
    rad = neuron_size_ls[i]
    membrane = Sphere_3d(center=Point(pos),
                         radius=rad)
    membrane_pht = Phantom(membrane, material=myelin)
    body = Sphere_3d(center=Point(pos),
                     radius=rad-2)
    body_pht = Phantom(body, material=water)
    nuclear = Sphere_3d(center=Point(pos),
                        radius=int(rad / 2))
    nuclear_pht = Phantom(nuclear, material=nucleic_acid)
    phantom.children.append(membrane_pht)
    phantom.children.append(body_pht)
    phantom.children.append(nuclear_pht)

print('Discretizing started.')
grid = discrete_phantom(phantom, 1,
                        bounding_box=[[0, grid_size[0]],
                                      [0, grid_size[1]],
                                      [0, grid_size[2]]],
                        prop=['delta', 'beta'],
                        ratio=1,
                        mkwargs={'energy': energy},
                        overlay_mode='replace')
grid_delta = grid[..., 0]
grid_beta = grid[..., 1]
print(grid_delta.shape)

np.save('neurons/grid_delta.npy', grid_delta)
np.save('neurons/grid_beta.npy', grid_beta)

# sim = Simulator(energy=25000,
#                 grid=(grid_delta, grid_beta),
#                 psize=[1.e-7, 1.e-7, 1.e-7])
#
# sim.initialize_wavefront('plane')
# t0 = time.time()
# wavefront = sim.multislice_propagate()
# print('Propagation time: {} ms'.format((time.time() - t0) * 1000))
#
# plt.imshow(np.abs(wavefront))


