import cmath
import numpy as np
EPSILON = 1e-5

def to_complex(vec2):
    return vec2[0] + vec2[1] * 1j

def complex_to_vec(c: complex) -> np.ndarray:
    return np.array([np.real(c), np.imag(c)])

def findMobiusTransform(z1, z2, z3, w1, w2, w3):
    """ inputs are vec2 (tuples) returns a mobius matrix 2x2 complex"""
    
    matA = np.array([
        [ to_complex(z1) * to_complex(w1), to_complex(w1), 1.0+0.0j],
        [ to_complex(z2) * to_complex(w2), to_complex(w2), 1.0+0.0j],
        [ to_complex(z3) * to_complex(w3), to_complex(w3), 1.0+0.0j]
    ])
    
    matB = np.array([
        [ to_complex(z1) * to_complex(w1), to_complex(z1), to_complex(w1)],
        [ to_complex(z2) * to_complex(w2), to_complex(z2), to_complex(w2)],
        [ to_complex(z3) * to_complex(w3), to_complex(z3), to_complex(w3)]
    ])
    
    matC = np.array([
        [ to_complex(z1), to_complex(w1), 1.0+0.0j],
        [ to_complex(z2), to_complex(w2), 1.0+0.0j],
        [ to_complex(z3), to_complex(w3), 1.0+0.0j]
    ])
    
    matD = np.array([
        [ to_complex(z1) * to_complex(w1), to_complex(z1), 1.0+0.0j],
        [ to_complex(z2) * to_complex(w2), to_complex(z2), 1.0+0.0j],
        [ to_complex(z3) * to_complex(w3), to_complex(z3), 1.0+0.0j]
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

    new_norm = a * d - b * c
    real = np.real(new_norm)
    imag = np.imag(new_norm)
    
    return np.array([
        [a, b],
        [c, d]
    ])

def transform(z: complex, mat):
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]
    
    return (a * z + b) / (c * z + d)

def exp_matrix(mat: np.ndarray) -> np.ndarray:
    # compute eigenvalues
    trace = mat[0][0] + mat[1][1]
    determinant = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
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
    # print('norm exp:', v1norm, v2norm)
    # if np.isnan(v1norm):
        # print(0)
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
    determinant = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
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
    # print('norm log:', v1norm, v2norm)
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
    mobius_ratio = np.matmul(a, np.linalg.inv(b))

    real = np.real(mobius_ratio)

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

def gamma_func(edge_i: complex, edge_j: complex, edge_k: complex, z: complex):
    r_ij = r_distance(edge_i, edge_j, z)
    r_ki = r_distance(edge_k, edge_i, z)
    r_jk = r_distance(edge_j, edge_k, z)

    s = (r_jk * r_ki) + (r_ij * r_jk) + (r_ij * r_ki)

    return (r_ij, r_ki, r_jk, s)

def gamma_ij(edge_i: complex, edge_j: complex, edge_k: complex, z: complex) -> float:
    r_ij, r_ki, r_jk, s = gamma_func(edge_i, edge_j, edge_k, z)

    if abs(s) < 0.00001:
        return 0.0
    return (r_jk * r_ki) / s

def gamma_jk(edge_i: complex, edge_j: complex, edge_k: complex, z: complex) -> float:
    r_ij, r_ki, r_jk, s = gamma_func(edge_i, edge_j, edge_k, z)

    if abs(s) < 0.00001:
        return 0.0
    return (r_ij * r_ki) / s

def gamma_ki(edge_i: complex, edge_j: complex, edge_k: complex, z: complex) -> float:
    r_ij, r_ki, r_jk, s = gamma_func(edge_i, edge_j, edge_k, z)

    if abs(s) < 0.00001:
        return 0.0
    return (r_ij * r_jk) / s

def log_ratio_interpolator_primary(edge_i: complex, edge_j: complex, edge_k: complex, t: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, z: complex) -> np.ndarray:
    g_ij = gamma_ij(edge_i, edge_j, edge_k, z)
    g_jk = gamma_jk(edge_i, edge_j, edge_k, z)
    g_ki = gamma_ki(edge_i, edge_j, edge_k, z)

    l_t = log_ratio_interpolator(u, t) * g_ij + log_ratio_interpolator(v, t) * g_jk + log_ratio_interpolator(w, t) * g_ki

    halved = l_t * 0.5

    interpolator_O = np.matmul(exp_matrix(halved), t) #mat multiply

    return interpolator_O

def log_ratio_interpolator_primary_and_transform(edge_i: complex, edge_j: complex, edge_k: complex, t: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, z: complex) -> complex:

    interpolator_O = log_ratio_interpolator_primary(edge_i, edge_j, edge_k, t, u, v, w, z)

    return transform(z, interpolator_O)

def test_mobius():
    mat = findMobiusTransform((-1,3), (6,9), (9,-1), (7, 3), (12,9), (12,1))
    
    assert(cmath.isclose(transform(-1+3j, mat), 7+3j))
    assert(cmath.isclose(transform(6+9j, mat), 12+9j))
    assert(cmath.isclose(transform(9-1j, mat), 12+1j))
    
def test_np_inv():
    matrix = np.array([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
            ])
    
    ad = (1+2j) * (7+8j)
    bc = (3+4j) * (5+6j)
    den = 1/(ad - bc)

    inv = den * np.array([
            [matrix[1][1], -matrix[0][1]],
            [-matrix[1][0], matrix[0][0]]
            ])
    
    inv_np = np.linalg.inv(matrix)

    assert np.allclose(inv_np, inv)

def test_multiply():
    a = np.array([
        [1, 2],
        [3, 4]
    ])

    b = np.array([
        [5, 4],
        [3, 2]
    ])

    c = a * b
    d = np.matmul(a, b)

def test_exp_mat():
    mat = np.array([
        [8, -7],
        [1, 0]
        ])
    exponented = exp_matrix(mat)

    symbolab_exponented = np.array([
        [1278.952305, -1276.234023],
        [182.3191461, -179.6008643]
        ])
    
    np_exponented = np.exp(mat)

    assert np.allclose(exponented, symbolab_exponented, 0.0001)
    assert not np.allclose(exponented, np_exponented, 0.0001)
    print(0)

def test_mobius_2():
    findMobiusTransform((465, 400), (375, 255), (285, 400), (0.721499979, 0.555000007), (0.5, 0.169499993), (0.277999997, 0.555000007))

def test_exp_mat_2():
    mat = np.array([
        [-0.0373644903+0.0475419611j, 0.0419622548-0.00615041098j], 
        [-0.0164517052+0.0920018032j, 0.0564881079-0.0366623886j]
    ])
    exp_matrix(mat)

def test_log_ratio_interpolator():
    a = np.array([
        [-6.91967216e-05 -6.33188506e-07j, 0.0115511687 +0.0121726356j],
        [-8.78454287e-08 -1.10134431e-06j, -0.0287222303 +0.000434322574j]
    ])

    b = np.array([
        [7.93345607e-05 -5.51156018e-06j, -0.0171067715 -0.0134160938j],
        [8.70759038e-08 -1.10524061e-05j, 0.0252380241 +0.00409506354j]
    ])

    log_ratio_interpolator(a, b)

def test_rdistance():
    r_distance(285+400j, 375+255j, 523+101j)

def test_gamma():
    gamma_ij(285+400j, 375+255j, 465+400j, 567+231j)

if __name__ == '__main__':
    
    test_mobius()
    test_np_inv()
    test_multiply()

    test_exp_mat()
    test_mobius_2()
    test_exp_mat_2()

    test_log_ratio_interpolator()
    test_rdistance()
    test_gamma()
    
    checkpoint = 0








