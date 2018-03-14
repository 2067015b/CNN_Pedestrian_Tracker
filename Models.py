import math

from condensation.ExtendedParticleFilter import ExtendedParticleFilter
from condensation.ParticleFilter import ParticleFilter
from Config import *


def particle_weight(state, position=(0, 0), velocity=(0, 0)):
    """ Updates the weight of one particle. """

    dist = math.sqrt(math.pow(state[0] - position[0], 2) +
                     math.pow(state[1] - position[1], 2) +
                     math.pow(state[2] - velocity[0], 2) +
                     math.pow(state[3] - velocity[1], 2))

    weight = math.exp(-dist / 10)

    return weight

class Trajectory(object):
    '''
    Class to represent the trajectory object.

    frames - an array with frame numbers where the pedestrian was detected
    coords - an array (indices correspond to indices in 'frames') with coordinate values of each detection
    '''

    trajectory_id = 0

    def __init__(self, frame_no, coords):
        Trajectory.trajectory_id += 1
        self.frames = [frame_no]
        self.coords = [coords]
        self.finished = False
        self.id = Trajectory.trajectory_id

        self.velocity = (0,0)
        num_particles = 500
        num_states = 4
        dynamics_matrix = [[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1]]
        lower_bounds = [0, 0, 0, 0]
        upper_bounds = [FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH,
                        FRAME_HEIGHT]
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
        pf.update(particle_weight,self.get_centroid(coords), self.velocity)

        self.particles = pf.particles
        self.pf = pf


    def __str__(self):
        return str(self.coords)

    def add_detection(self, frame_no, coords):
        self.frames.append(frame_no)
        self.update_velocity(coords, self.coords[-1])
        self.coords.append(coords)
        self.pf.update(particle_weight,self.get_centroid(coords), self.velocity)
        self.particles = self.pf.particles

    def update_velocity(self, new, old):
        self.velocity = (new[0] - old[0], new[1] - old[1])

    def get_detections(self, frame=None):
        i = -1
        if frame:
            try:
                i = self.frames.index(frame)
            except:
                pass
        return self.coords[i]

    def get_centroid(self, coords):
        return coords[0] + ((coords[2] - coords[0]) / 2), coords[1] + ((coords[3] - coords[1]) / 2)

class Detection(object):
    '''
    Class to represent a single detection.

    coords - an array with x, y coordinates of the bounding box
    '''

    def __init__(self, coords):
        self.coords = coords

    def __str__(self):
        return str(self.coords)

    def get_coordinates(self):
        return self.coords

    def get_centroid(self):
        return self.coords[0] + ((self.coords[2] - self.coords[0]) / 2), self.coords[1] + ((self.coords[3] - self.coords[1]) / 2)