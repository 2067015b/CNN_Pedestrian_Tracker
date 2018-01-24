###############################################################################
#
# Copyright (c) 2016, Henrique Morimitsu,
# University of Sao Paulo, Sao Paulo, Brazil
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# #############################################################################

import numpy as np
import random


class ParticleFilter(object):
    """ Implements a particle filter using ConDensation, as proposed in:
    Michael Isard and Andrew Blake. CONDENSATION - conditional
    density propagation for visual tracking. International journal of computer
    vision, 29(1): 5-28.
    """
    def __init__(self, num_particles, num_states, dynamics_matrix,
                 particle_lower_bounds, particle_upper_bounds,
                 noise_type='gaussian', noise_param1=None, noise_param2=None,
                 final_state_decision_method='weighted_average'):
        """ dynamics_matrix is a ns x ns square matrix, where ns = num_states
        particle_lower_bounds is a vector that represents
            the minimum values of each state
        particle_upper_bounds is a vector that represents
            the maximum values of each state
        noise_type must be either 'gaussian' or 'uniform'
        noise_param1 must be either None of a vector with num_states elements.
            If it is set as None, then it is initialized as a vector of zeros.
            If noise_type is gaussian, this parameter represents the means of
            the noise distribution, while if the noise_type is uniform, then
            it represents the lower bounds of the interval
        noise_param2 is similar to noise_param1. If it is None, it is set as a
            vector of ones. When the noise_type is gaussian, it represents the
            standard deviations, while if it is uniform, it is the upper bounds
            of the interval
        final_state_decision_method must be either 'best', 'average' or
            'weighted_average'. If best, the particle with highest weight is
            chosen as the new state. If average, the new state is computed
            from the simple average of all the particles. If weighted_average,
            the state comes from an average of all particles averaged by
            their weights
        """
        self._num_particles = num_particles
        self._num_states = num_states
        self._dynamics_matrix = np.array(dynamics_matrix)
        self._particle_lower_bounds = np.array(particle_lower_bounds)
        self._particle_upper_bounds = np.array(particle_upper_bounds)
        self._noise_type = noise_type
        if noise_param1 is None:
            self._noise_param1 = num_states * [0.0]
        elif len(noise_param1) == num_states:
            self._noise_param1 = noise_param1
        if noise_param2 is None:
            self._noise_param2 = num_states * [1.0]
        elif len(noise_param2) == num_states:
            self._noise_param2 = noise_param2
        self._final_state_decision_method = final_state_decision_method
        self._final_state = num_states * [0.0]
        self._particles = np.zeros((num_particles, num_states), np.float64)
        self._weights = np.zeros((num_particles), np.float64)
        self._normalized_weights = np.zeros((num_particles), np.float64)
        self._weight_sum = 0.0
        self._cumulative_weights = np.zeros((num_particles), np.float64)

        self._init_weights()

    def _apply_dynamics(self, state):
        """ Applies dynamics to one particle, according to
        self.dynamics_matrix.
        """
        return np.dot(state, self._dynamics_matrix)

    def _apply_noise(self, state):
        """ Applies noise to one particle, according to self.noise_type """
        noise = np.zeros((self._num_states), np.float64)
        for j in range(len(state)):
            if self._noise_type == 'uniform':
                noise[j] = random.uniform(self._noise_param1[j],
                                          self._noise_param2[j])
            if self._noise_type == 'gaussian':
                noise[j] = random.gauss(self._noise_param1[j],
                                        self._noise_param2[j])
        return state + noise

    def get_final_state(self):
        """ Computes the final state estimated by the particles, according to
        self.final_state_decision_method. Returns the final state itself and
        also the sum of weights of the particles that generate it.
        """
        total_weight = 0
        if self._final_state_decision_method == 'best':
            index = 0
            highest_weight = 0.0
            for i in range(self._num_particles):
                if self._weights[i] > highest_weight:
                    highest_weight = self._weights[i]
                    index = i
            final_state = self._particles[index].copy()
            total_weight = highest_weight
        elif self._final_state_decision_method == 'average':
            final_state = np.zeros((self._num_states), np.float64)
            for i in range(self._num_particles):
                final_state += self._particles[i]
            final_state /= self._num_particles
            total_weight = self._weight_sum
        elif self._final_state_decision_method == 'weighted_average':
            final_state = np.zeros((self._num_states), np.float64)
            if self._weight_sum > 0:
                for i in range(self._num_particles):
                    final_state += self._particles[i] * \
                        self._weights[i] / self._weight_sum
            total_weight = self._weight_sum

        self._final_state = self._apply_dynamics(final_state)

        return self._final_state, total_weight

    def init_particles(self, init_method='uniform',
                       init_param1=None, init_param2=None):
        """ Initialize all the particles.
        init_method must be either 'uniform' or 'gaussian'. This parameter
            indicates how the particles are initially spread in the state space
        init_param1 must be either None of a vector with self.num_states
            elements. If it is set as None, then it is initialized as
            self.particle_lower_bounds. If noise_type is gaussian, this
            parameter represents the means of the noise distribution, while if
            the noise_type is uniform, then it represents the lower bounds of
            the interval
        init_param2 is similar to init_param2. If it is None, it is set as
            self.particle_upper_bounds. When the noise_type is gaussian, it
            represents the standard deviations, while if it is uniform, it is
            the upper bounds of the interval
        """
        for i in range(self._num_particles):
            particle_state = np.zeros((self._num_states), np.float64)

            for j in range(self._num_states):
                if init_method == 'uniform':
                    if init_param1 is None or init_param2 is None:
                        particle_state[j] = \
                            random.uniform(self._particle_lower_bounds[j],
                                           self._particle_upper_bounds[j])
                    else:
                        particle_state[j] = random.uniform(init_param1[j],
                                                           init_param2[j])
                elif init_method == 'gaussian':
                    if init_param1 is None or init_param2 is None:
                        particle_state[j] = \
                            random.gauss(self._particle_lower_bounds[j],
                                         self._particle_upper_bounds[j])
                    else:
                        particle_state[j] = random.gauss(init_param1[j],
                                                         init_param2[j])

            self._particles[i] = particle_state

        self._final_state = self.get_final_state()

    def _init_weights(self):
        """ Initialize the weights of the particles with
        a uniform distribution.
        """
        weight = 1.0 / self._num_particles
        self._weights += weight
        self._normalized_weights += weight

        self._cumulative_weights = np.arange(weight, 1.0 + weight, weight,
                                             np.float64)
        self._weight_sum = 1.0

    def _propagate_particles(self):
        """ Applies dynamics and noise to all the particles. """
        for i in range(self._num_particles):
            self._particles[i] = self._update_particle(self._particles[i])

    def _resample_particles(self):
        """ Resample new particles from the old ones, according to
        self.resampling_method.
        """
        old_particles = self._particles.copy()
        old_weights = self._weights.copy()
        old_normalized_weights = self._normalized_weights.copy()
        for i in range(self._num_particles):
            x = random.uniform(0.0, self._weight_sum)
            j = np.searchsorted(self._cumulative_weights, x)
            self._particles[i] = old_particles[j].copy()
            self._weights[i] = old_weights[j].copy()
            self._normalized_weights[i] = old_normalized_weights[j].copy()

    def update(self, weighting_function, *args):
        """ Updates all the particles by resampling, propagating and updating
        their weights.
        """
        self._resample_particles()
        self._propagate_particles()
        self._update_weights(weighting_function, *args)

    def _update_particle(self, state):
        """ Applies dynamics and noise to one particle. """
        new_state = self._apply_dynamics(state)
        new_state = self._apply_noise(new_state)

        return new_state

    def _update_weights(self, weighting_function, *args):
        """ Updates the weight of all the particles.
        weighting_function is a reference to a function that effectively
        computes the new weights of the particles *args are the parameters,
        besides the particle state, that weighting_function may require
        """
        for i in range(len(self._particles)):
            self._weights[i] = weighting_function(self._particles[i], *args)

        self._weight_sum = np.sum(self._weights)
        self._cumulative_weights = np.cumsum(self._weights)

        if self._weight_sum > 0:
            for i in range(len(self._particles)):
                self._normalized_weights[i] = \
                    self._weights[i] / self._weight_sum

    @property
    def normalized_weights(self):
        return self._normalized_weights

    @property
    def num_states(self):
        return self._num_states

    @property
    def particles(self):
        return self._particles

    @property
    def weights(self):
        return self._weights

    @property
    def weight_sum(self):
        return self._weight_sum
