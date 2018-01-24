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

from ParticleFilter import ParticleFilter
import math

""" This sample application creates a series of numbers and then uses
particle filter to try to predict the next number of the sequence.
"""


def create_series():
    """ Creates a sequence of numbers for testing the tracking. """
    forward_interval = 20
    backward_interval = -10
    maxval = 200
    series = []
    for i in range(forward_interval, maxval, forward_interval):
        series.append((i, forward_interval))
    for i in range(maxval, 0, backward_interval):
        series.append((i, backward_interval))

    return series


def init_particle_filter():
    """ Initialize the particle filter. """
    num_particles = 500
    num_states = 2
    dynamics_matrix = [[1, 0],
                       [1, 1]]
    lower_bounds = [0, 0]
    upper_bounds = [200, 200]
    noise_type = 'gaussian'
    noise_param1 = num_states * [0.0]
    noise_param2 = num_states * [5.0]
    final_state_decision_method = 'weighted_average'
    pf = ParticleFilter(num_particles, num_states, dynamics_matrix,
                        lower_bounds, upper_bounds,
                        noise_type, noise_param1, noise_param2,
                        final_state_decision_method)
    particle_init_method = 'uniform'
    pf.init_particles(particle_init_method, lower_bounds, upper_bounds)

    return pf


def particle_weight(state, x, dx):
    """ Updates the weight of one particle. """
    dist = math.sqrt(math.pow(state[0] - x, 2) + math.pow(state[1] - dx, 2))

    return 1.0 / dist


def main():
    series = create_series()
    pf = init_particle_filter()

    print('actual_number/interval\tpredicted_next_number/interval')
    for x in series:
        pf.update(particle_weight, x[0], x[1])
        final_state = pf.get_final_state()[0]

        print(str(x) + '\t\t(' + str(final_state[0]) + ', ' +
              str(final_state[1]) + ')')

if __name__ == "__main__":
    main()
