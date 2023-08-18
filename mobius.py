import cmath
import numpy as np
EPSILON = 1e-5

def findMobiusTransform(z1, z2, z3, w1, w2, w3):
    """ inputs are vec2 (tuples) returns a mobius matrix 2x2 complex"""
    
    def to_complex(vec2):
        return vec2[0] + vec2[1] * 1j
    
    matA = np.array([
        [ to_complex(z1) * to_complex(w1), to_complex(w1), 1.0],
        [ to_complex(z2) * to_complex(w2), to_complex(w2), 1.0],
        [ to_complex(z3) * to_complex(w3), to_complex(w3), 1.0]
    ])
    
    matB = np.array([
        [ to_complex(z1) * to_complex(w1), to_complex(z1), to_complex(w1)],
        [ to_complex(z2) * to_complex(w2), to_complex(z2), to_complex(w2)],
        [ to_complex(z3) * to_complex(w3), to_complex(z3), to_complex(w3)]
    ])
    
    matC = np.array([
        [ to_complex(z1), to_complex(w1), 1.0],
        [ to_complex(z2), to_complex(w2), 1.0],
        [ to_complex(z3), to_complex(w3), 1.0]
    ])
    
    matD = np.array([
        [ to_complex(z1) * to_complex(w1), to_complex(z1), 1.0],
        [ to_complex(z2) * to_complex(w2), to_complex(z2), 1.0],
        [ to_complex(z3) * to_complex(w3), to_complex(z3), 1.0]
    ])
    
    a = np.linalg.det(matA)
    b = np.linalg.det(matB)
    c = np.linalg.det(matC)
    d = np.linalg.det(matD)
    
    norm = a * d - b * c
    
    a /= norm
    b /= norm
    c /= norm
    d /= norm
    
    return np.array([
        [a, b],
        [c, d]
    ])

def transform(z, mat):
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]
    
    return (a * z + b) / (c * z + d)

def test_mobius():
    mat = findMobiusTransform((-1,3), (6,9), (9,-1), (7, 3), (12,9), (12,1))
    
    assert(cmath.isclose(transform(-1+3j, mat), 7+3j))
    assert(cmath.isclose(transform(6+9j, mat), 12+9j))
    assert(cmath.isclose(transform(9-1j, mat), 12+1j))
    

if __name__ == '__main__':
    
    test_mobius()






