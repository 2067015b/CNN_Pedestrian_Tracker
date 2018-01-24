Particle Filter Tracking
------------------------

This code implements particle filter to track using the ConDensation algorithm.
It also contains some sample applications to demonstrate how the particle
filter may be used to track.

Copyright (c) 2016, Henrique Morimitsu,
University of Sao Paulo, Sao Paulo, Brazil

Contact: henriquem87@gmail.com


Requirements
------------

Python: https://www.python.org/

Numpy: http://www.numpy.org/

For some of the sample applications:

OpenCV: http://opencv.org/

This code was implemented and tested using Python 3.4 with OpenCV 3.0.0
but it should also work with Python 2.X and OpenCV 2.4.X

Usage
-----

The three sample application are:

- trackSeries.py: predicts the next number in a sequence

- trackMouse.py: tracks the mouse pointer inside a window

- trackVideo.py: tracks one object in a video

They can be executed by calling the python interpreter. For example:
> python trackSeries.py

