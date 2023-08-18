import numpy as np
import math

def perspective(fovy, aspect, z_near, z_far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    return np.array([
        [f / aspect,  0,                                   0,  0],
        [          0, f,                                   0,  0],
        [          0, 0, (z_far + z_near) / (z_near - z_far), -1],
        [          0, 0, (2*z_far*z_near) / (z_near - z_far),  0]
    ])

def translate(tx, ty, tz):
    return np.array([
        [ 1, 0, 0, tx],
        [ 0, 1, 0, ty],
        [ 0, 0, 1, tz],
        [ 0, 0, 0, 1]
    ])

def scale(value):
    return np.array([
        [ value, 0, 0, 0],
        [ 0, value, 0, 0],
        [ 0, 0, value, 0],
        [ 0, 0, 0, 1]
    ])
    
def create_identity():
        return np.array([
        [ 1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1]
    ])

def rotate(angle, x, y, z):
    s = math.sin(math.radians(angle))
    c = math.cos(math.radians(angle))
    magnitude = math.sqrt(x*x + y*y + z*z)
    nc = 1 - c
      
    x /= magnitude
    y /= magnitude
    z /= magnitude

    return np.array([
        [     c + x**2 * nc, y * x * nc - z * s, z * x * nc + y * s, 0],
        [y * x * nc + z * s,      c + y**2 * nc, y * z * nc - x * s, 0],
        [z * x * nc - y * s, z * y * nc + x * s,      c + z**2 * nc, 0],
        [                 0,                  0,                  0, 1],
    ])