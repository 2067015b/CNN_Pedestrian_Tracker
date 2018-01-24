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

import cv2
from ExtendedParticleFilter import ExtendedParticleFilter
import numpy as np
import math

""" This sample application uses particle filter to track the mouse pointer
inside a window.
"""

mouse_position = (0, 0)
velocity = (0, 0)


def init_particle_filter(window_size):
    """ Initialize the particle filter. """
    num_particles = 500
    num_states = 4
    dynamics_matrix = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [1, 0, 1, 0],
                       [0, 1, 0, 1]]
    lower_bounds = [0, 0, 0, 0]
    upper_bounds = [window_size[0], window_size[1], window_size[0],
                    window_size[1]]
    noise_type = 'gaussian'
    noise_param1 = num_states * [0.0]
    noise_param2 = num_states * [5.0]
    maximum_total_weight = 1.0 * num_particles
    final_state_decision_method = 'weighted_average'
    noise_dispersion_based_on_weight = True
    dispersion_factor = 2.0
    minimum_dispersion = 0.2
    pf = ExtendedParticleFilter(
        num_particles, num_states, dynamics_matrix, lower_bounds, upper_bounds,
        noise_type, noise_param1, noise_param2, maximum_total_weight,
        final_state_decision_method, noise_dispersion_based_on_weight,
        dispersion_factor, minimum_dispersion)
    particle_init_method = 'uniform'
    pf.init_particles(particle_init_method, lower_bounds, upper_bounds)

    return pf


def on_mouse(event, x, y, flags, param):
    """ Mouse callback function. It is called from the OpenCV windows
    to obtain the mouse values. """
    global mouse_position
    global velocity

    velocity = (x - mouse_position[0], y - mouse_position[1])
    mouse_position = (x, y)


def particle_weight(state):
    """ Updates the weight of one particle. """
    global mouse_position
    global velocity

    dist = math.sqrt(math.pow(state[0] - mouse_position[0], 2) +
                     math.pow(state[1] - mouse_position[1], 2) +
                     math.pow(state[2] - velocity[0], 2) +
                     math.pow(state[3] - velocity[1], 2))

    weight = math.exp(-dist / 10)

    return weight


def main():
    window_size = (640, 480)
    pf = init_particle_filter(window_size)

    # Create a black image, a window and bind the function to the window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)

    while(True):
        img = np.zeros((window_size[1], window_size[0], 3), np.uint8)
        particles = pf.particles

        for p in particles:
            x, y = p[:2]
            cv2.circle(img, (int(x), int(y)), 7, (255, 0, 0), 1)

        pf.update(particle_weight)
        final_state = pf.get_final_state()[0]
        x, y = final_state[:2]
        cv2.circle(img, (int(x), int(y)), 7, (0, 0, 255), -1)

        cv2.imshow('image', img)

        key = cv2.waitKey(20)
        if key % 256 == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
