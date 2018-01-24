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
from condensation.Rectangle import Rectangle
from condensation.ColorObjectPFTracker import ColorObjectPFTracker

""" This sample application uses particle filter to track an object in a video
using a color histogram model. The user can draw a bounding box in the beginning
of the video, which will be used to learn the target model. This application
can use both a video or a webcam, depending on the execution parameters.
"""

# Auxiliary variables to draw to initial bounding box around the target
box_drawn = False
model_built = False
color_tracker = None


def draw_particles(img, objBB, particles, final_state,
                   particles_color=(255, 0, 0),
                   final_state_color=(0, 0, 255)):
    """ Draws the particles and tracked bouding box on the screen. """
    num_particles = len(particles)
    # The interval is used to avoid drawing more than 50 particles
    interval = num_particles / 50

    for i, p in enumerate(particles):
        if i % interval == 0:
            particleBB = objBB.centered_on(p[0], p[1])
            cv2.rectangle(img, particleBB.tl(), particleBB.br(),
                          particles_color, 1)

    final_stateBB = objBB.centered_on(final_state[0], final_state[1])
    cv2.rectangle(img, final_stateBB.tl(), final_stateBB.br(),
                  final_state_color, 3)

    return img


def init_particle_filter(window_size, lower_bounds, upper_bounds):
    """ Initialize the particle filter. """
    num_particles = 50
    num_states = 2
    dynamics_matrix = [[1, 0],
                       [0, 1]]
    noise_type = 'gaussian'
    noise_param1 = num_states * [0.0]
    noise_param2 = num_states * [5.0]
    final_state_decision_method = 'weighted_average'
    maximum_total_weight = 0.5 * num_particles
    noise_dispersion_based_on_weight = True
    dispersion_factor = 5.0
    minimum_dispersion = 0.5
    color_tracker = ColorObjectPFTracker(
        num_particles, num_states, dynamics_matrix, lower_bounds, upper_bounds,
        noise_type, noise_param1, noise_param2, maximum_total_weight,
        final_state_decision_method, noise_dispersion_based_on_weight,
        dispersion_factor, minimum_dispersion)

    return color_tracker


def is_cv2():
    return cv2.__version__.startswith('2.')


def is_cv3():
    return cv2.__version__.startswith('3.')



def get_predictions_for_frame(frame_0, frame_1, detections):
    global model_built
    global color_tracker

    # The frame is rescaled to have the width of 640 pixels
    frame_width = 640
    rescale_factor = float(frame_width) / frame_0.shape[1]
    frame_height = int(frame_0.shape[0] * rescale_factor)

    if not color_tracker:
        lower_bounds = [0, 0]
        upper_bounds = [frame_width, frame_height]
        color_tracker = init_particle_filter((frame_width, frame_height),
                                             lower_bounds, upper_bounds)

    # Variables for computing the color histogram model
    channels = [0, 1, 2]
    mask = None
    numBins = [8, 8, 4]
    intervals = [0, 180, 0, 256, 0, 256]

    particle_initialization_method = 'gaussian'

    output_frame = cv2.resize(frame_0, (frame_width, frame_height))

    predictions = []

    for detection in detections:


        model_objectBB = Rectangle(
            detection[0], detection[1],
            detection[2] - detection[0],
            detection[3] - detection[1])

        # The model is only learned once in the beginning
        if not model_built:
            hsv_img = cv2.cvtColor(output_frame, cv2.COLOR_BGR2HSV)


            color_tracker.init_color_particles(
                particle_initialization_method, model_objectBB.centroid(),
                [5.0, 5.0])
            color_tracker.init_object_histogram_model(
                hsv_img, model_objectBB, channels, mask, numBins,
                intervals)

            model_built = True

        # Grabs next frame
        # frame_0 = cap.read()[1]

        output_frame = cv2.resize(frame_1, (frame_width, frame_height))
        hsv_img = cv2.cvtColor(output_frame, cv2.COLOR_BGR2HSV)

        final_state = color_tracker.update_tracking(hsv_img)[0]
        particle_centres = color_tracker.particles

        num_particles = len(particle_centres)
        # The interval is used to avoid drawing more than 50 particles
        interval = num_particles / 50

        particles = []
        for i, p in enumerate(particle_centres):
            if i % interval == 0:
                particles.append(model_objectBB.centered_on(p[0], p[1]).get_coords())

        predictions.append([model_objectBB.centered_on(final_state[0], final_state[1]).get_coords(),particles])

    return predictions
