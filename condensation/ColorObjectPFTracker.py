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
import math
import numpy as np
from condensation.ExtendedParticleFilter import ExtendedParticleFilter
from condensation.Rectangle import Rectangle


class ColorObjectPFTracker(object):
    """ This class implements a visual tracker based on color histogram with
    particle filter. """
    def __init__(self, num_particles, num_states, dynamics_matrix,
                 particle_lower_bounds, particle_upper_bounds,
                 noise_type='gaussian', noise_param1=None, noise_param2=None,
                 maximum_total_weight=0.0,
                 final_state_decision_method='weighted_average',
                 noise_dispersion_based_on_weight=False, dispersion_factor=5.0,
                 minimum_dispersion=0.5):
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
        maximum_total_weight is the highest value that it is expected to be
            obtained by summing all the weights. This parameter is necessary if
            the weighting function is not bounded or if the real maximum value
            is never reached in real situations. This value should be set as the
            highest value that usually occurs
        final_state_decision_method must be either 'best', 'average' or
            'weighted_average'. If best, the particle with highest weight is
            chosen as the new state. If average, the new state is computed
            from the simple average of all the particles. If weighted_average,
            the state comes from an average of all particles averaged by
            their weights
        noise_dispersion_based_on_weight if set as False, then this class
            behaves the same way as the ParticleFilter parent class
        dispersion_factor adjusts the variance of the noise, the higher the
            dispersion_factor, the higher the variance.
        minimum_dispersion is the lower bound of the noise dispersion
        """
        self._pf = ExtendedParticleFilter(
            num_particles, num_states, dynamics_matrix, particle_lower_bounds,
            particle_upper_bounds, noise_type, noise_param1, noise_param2,
            maximum_total_weight, final_state_decision_method,
            noise_dispersion_based_on_weight, dispersion_factor,
            minimum_dispersion)
        self._modelBB = Rectangle(0, 0, 0, 0)
        self._model_hist_params = None
        self._model_hist = None

    def compute_object_histogram(self, hsv_img, objectBB, channels, mask, num_bins,
                                 intervals):
        """ Computes the color histogram of a bounding box.
        The color model corresponds to the method proposed in:
        Patrick Perez, Carine Hue, Jaco Vermaak and Michel Gangnet.
        Color-based probabilistic tracking. In European Conference on Computer
        Vision, pages 661-675. Springer.
        """
        obj_hist = np.zeros((num_bins[0] * num_bins[1] + num_bins[2]),
                            np.float32)
        obj_image = hsv_img[objectBB.top():objectBB.bottom(),
                        objectBB.left():objectBB.right()]

        # Creates a separated image for each channel
        splitted_img = cv2.split(obj_image)
        if len(splitted_img) == 3:
            maskH = cv2.threshold(splitted_img[0], int(0.1 * 255), 255,
                                  cv2.THRESH_BINARY)[1]
            maskS = cv2.threshold(splitted_img[1], int(0.2 * 255), 255,
                                  cv2.THRESH_BINARY)[1]
            maskHS = cv2.bitwise_and(maskH, maskS)
            white_mask = np.ones_like(maskH) * 255
            maskV = cv2.bitwise_xor(maskHS, white_mask)
            hs_hist = cv2.calcHist([obj_image], channels[:2], maskHS,
                                   num_bins[:2], intervals[:4])
            v_hist = cv2.calcHist([obj_image], channels[2:3], maskV,
                                  num_bins[2:3], intervals[4:])
            obj_hist = np.concatenate((hs_hist.flatten(), v_hist.flatten()))
            obj_hist /= np.sum(obj_hist)

        return obj_hist

    def init_color_particles(self, init_method='uniform', init_param1=None,
                             init_param2=None):
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
        self._pf.init_particles(init_method, init_param1, init_param2)

    def init_object_histogram_model(self, img, objectBB, channels, mask,
                                    num_bins, intervals):
        """ Compute the model histogram from the initial bounding box.
        img is a colored image matrix
        objectBB is the bounding box from where the histogram will be computed
        channels is a list of the indices of the channels that will be used
        mask is a binary image that represents the are that should be considered
            from the image
        num_bins is a list of how many bins each channels will have
        intervals is a list of lower and upper bounds for the values in each
            channel

        For more information about the last 4 parameters, consult the OpenCV
        documentation about the calcHist function.
        """
        self._modelBB = objectBB
        self._model_hist_params = [channels, mask, num_bins, intervals]
        self._model_hist = self.compute_object_histogram(
            img, objectBB, *self._model_hist_params)

    def particle_weight(self, state, hsv_img, channels, mask, num_bins, intervals):
        """ Computes the new weight of a particle. This function computes the
        likelihood P(z|x), where z is the observation (color histogram) and x
        the state. This implementations corresponds to the function proposed in:
        Erkut Erdem, Severine Dubuisson and Isabelle Bloch.
        Fragments based tracking with adaptive cue integration. Computer Vision
        and Image Understanding, 116 (7):827-841.
        """
        particleBB = self._modelBB.centered_on(state[0], state[1])
        particle_hist = self.compute_object_histogram(
            hsv_img, particleBB, channels, mask, num_bins, intervals)

        dist = cv2.compareHist(self._model_hist, particle_hist,
                               cv2.HISTCMP_HELLINGER)

        sigma = 0.1
        weight = 0.0
        if 0 <= state[0] < hsv_img.shape[1] and 0 <= state[1] < hsv_img.shape[0]:
            weight = math.exp(-(math.pow(dist, 2)) / (2 * math.pow(sigma, 2)))

        return weight

    def update_object_histogram(self, new_histogram, update_factor=0.1):
        """ Compute a linear combination of the current model
        histogram with another one.
        """
        self._model_hist = (1.0 - update_factor) * self._model_hist + \
            update_factor * new_histogram

    def update_tracking(self, hsv_img):
        """ Updates the particle filter and returns the next predicted state and
        the total weights of the particles.
        """
        self._pf.update(self.particle_weight, hsv_img, *self._model_hist_params)
        final_state, total_weight = self._pf.get_final_state()

        return final_state, total_weight

    @property
    def num_states(self):
        return self._pf.num_states

    @property
    def particles(self):
        return self._pf.particles

    @property
    def modelBB(self):
        return self._modelBB

    @property
    def weight_sum(self):
        return self._pf.weight_sum
