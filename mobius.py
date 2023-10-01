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

def mobius_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ''' a, b are complex matrix 2x2 UNCHEKED '''
    return a * np.linalg.inv(b)

def exp_matrix(mat: np.ndarray) -> np.ndarray:
    # compute eigenvalues
    trace = mat[0][0] + mat[1][1]
    determinant = mat[0][0] * mat[1][1] - max[0][1] * mat[1][0]
    discriminant = np.sqrt(trace * trace - 4.0 * determinant)
    lambda1 = (trace - discriminant) / 2.0
    lambda2 = (trace + discriminant) / 2.0

    # compute eigenvectors
    v1x = 1.0 + 0.0j
    v1y = (lambda1 - mat[0][0]) / mat[0][1]
    v2x = 1.0 + 0.0j
    v2y = (lambda2 - mat[0][0]) / mat[0][1]

    # normalize eigenvectors
    v1norm = np.sqrt(v1x * v1x + v1y * v1y)
    v2norm = np.sqrt(v2x * v2x + v2y * v2y)
    v1x /= v1norm
    v1y /= v1norm
    v2x /= v2norm
    v2y /= v2norm

    # diagonalize the matrix
    d_exp = np.array([
        [np.exp(lambda1), 0.0 + 0.0j],
        [0.0 + 0.0j, np.exp(lambda2)]
    ])

    V = np.array([
        [v1x, v2x],
        [v1y, v2y]
    ])

    result = np.matmul(np.matmul(V, d_exp), np.linalg.inv(V))
    return result

def log_matrix(mat: np.ndarray) -> np.ndarray:
    # compute eigenvalues
    trace = mat[0][0] + mat[1][1]
    determinant = mat[0][0] * mat[1][1] - max[0][1] * mat[1][0]
    discriminant = np.sqrt(trace * trace - 4.0 * determinant)
    lambda1 = (trace - discriminant) / 2.0
    lambda2 = (trace + discriminant) / 2.0

    # compute eigenvectors
    v1x = 1.0 + 0.0j
    v1y = (lambda1 - mat[0][0]) / mat[0][1]
    v2x = 1.0 + 0.0j
    v2y = (lambda2 - mat[0][0]) / mat[0][1]

    # normalize eigenvectors
    v1norm = np.sqrt(v1x * v1x + v1y * v1y)
    v2norm = np.sqrt(v2x * v2x + v2y * v2y)
    v1x /= v1norm
    v1y /= v1norm
    v2x /= v2norm
    v2y /= v2norm

    # diagonalize the matrix
    d_log = np.array([
        [np.log(lambda1), 0.0 + 0.0j],
        [0.0 + 0.0j, np.log(lambda2)]
    ])

    V = np.array([
        [v1x, v2x],
        [v1y, v2y]
    ])

    result = np.matmul(np.matmul(V, d_log), np.linalg.inv(V))
    return result

def log_ratio_interpolator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mobius_ratio = a * np.linalg.inv(b)

    real = np.real(mobius_ratio) # UNCHEKED

    trace = np.real(np.trace(real))

    sign_of_trace = np.sign(trace)

    log_arg = mobius_ratio * sign_of_trace

    result = log_matrix(log_arg)

    return result

def r_distance(edge1: complex, edge2: complex, z: complex) -> float:
    x0 = np.real(z)
    y0 = np.imag(z)
    x1 = np.real(edge1)
    y1 = np.imag(edge1)
    x2 = np.real(edge2)
    y2 = np.imag(edge2)

    up = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    down = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    return up / down


def test_mobius():
    mat = findMobiusTransform((-1,3), (6,9), (9,-1), (7, 3), (12,9), (12,1))
    
    assert(cmath.isclose(transform(-1+3j, mat), 7+3j))
    assert(cmath.isclose(transform(6+9j, mat), 12+9j))
    assert(cmath.isclose(transform(9-1j, mat), 12+1j))
    

if __name__ == '__main__':
    
    test_mobius()






