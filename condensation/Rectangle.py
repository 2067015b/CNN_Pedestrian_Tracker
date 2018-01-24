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


class Rectangle(object):
    """ Represents a rectangle. """

    def __init__(self, x, y, width, height):
        self._x = int(x)
        self._y = int(y)
        self._width = int(width)
        self._height = int(height)

    def bottom(self):
        return self._y + self._height

    def br(self):
        """ Returns the coordinate of the bottom-right corner. """
        return (self._x + self._width, self._y + self._height)

    def centered_on(self, x, y):
        """ Returns a rectangle with self.width and self.height
        centered on (x, y)"""
        return Rectangle(x - self._width / 2, y - self._height / 2,
                         self._width, self._height)

    def centroid(self):
        """ Returns the coordinate of the centroid. """
        return (self._x + self._width / 2, self._y + self._height / 2)

    def left(self):
        return self._x

    def right(self):
        return self._x + self._width

    def tl(self):
        """ Returns the coordinate of the top-left corner. """
        return (self._x, self._y)

    def tlbr(self):
        """ Returns a representation of the rectangle using the top-left and
        bottom-right corners. """
        return (self.tl(), self.br())

    def top(self):
        return self._y

    def get_coords(self):
        return list(self.tl()) + list(self.br())

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
