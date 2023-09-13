import numpy as np

def rotate_to_align(triangle, neighbor):
    t = triangle
    t1 = neighbor
    
    # find flat_lim
    flat_lim = 101010
    if t1[0][2] == t1[1][2]:
        flat_lim = t1[1][2]
    elif t1[0][2] == t1[2][2]:
        flat_lim = t1[2][2]
    else:
        flat_lim = t1[1][2]
    assert(flat_lim != 101010)
    
    # find vertex to rotate
    vertex_to_rotate = None
    vertex_to_rotate_index = None
    for index, i in enumerate(t1):
        if i[2] == flat_lim:
            pass
        else:
            vertex_to_rotate = i
            vertex_to_rotate_index = index
            
    assert(vertex_to_rotate is not None)
    # print(vertex_to_rotate, flat_lim)
    
    t1[vertex_to_rotate_index] = flat_lim
    return t1


def rotate_to_align_not_working(triangle, neighbor):

    t = triangle
    t1 = neighbor
    
    v1 = t[1] - t[0]
    v2 = t[2] - t[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal) # normal should be z
    
    # print(t1)
    # find flat_lim
    flat_lim = 101010
    if t1[0][2] == t1[1][2]:
        flat_lim = t1[1][2]
    elif t1[0][2] == t1[2][2]:
        flat_lim = t1[2][2]
    else:
        flat_lim = t1[1][2]
    assert(flat_lim != 101010)

    # find the axis - two vertices on the target triangle that are not at flat_lim
    axis = []
    axis_index = []
    vertex_to_rotate = None
    vertex_to_rotate_index = None
    for index, i in enumerate(t1):
        # print(i[2], flat_lim)
        if i[2] == flat_lim:
            axis.append(i)
            axis_index.append(index)
        else:
            vertex_to_rotate = i
            vertex_to_rotate_index = index
            
    assert(len(axis) == 2)
    assert(vertex_to_rotate is not None)
    # print(axis)

    v1 = t1[1] - t1[0]
    v2 = t1[2] - t1[0]
    neighbor_normal = np.cross(v1, v2)
    neighbor_normal = neighbor_normal / np.linalg.norm(neighbor_normal)

    # print(neighbor_normal)

    cos_angle = np.dot(normal, neighbor_normal) / (np.linalg.norm(normal) * np.linalg.norm(neighbor_normal))
    # print(cos_angle)
    angle = -np.arccos(np.clip(cos_angle, -1, 1))
    # print(angle)

    rotation_axis = (axis[1] - axis[0]) / np.linalg.norm(axis[1] - axis[0])
    # print(rotation_axis)

    cross_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                             [rotation_axis[2], 0, -rotation_axis[0]],
                             [-rotation_axis[1], rotation_axis[0], 0]])
                             
    rotation_matrix = np.identity(3) + np.sin(angle) * cross_matrix + (1 - np.cos(angle)) * np.dot(cross_matrix, cross_matrix)
    rotated_vertex = np.dot(rotation_matrix, vertex_to_rotate - axis[0]) + axis[0]

    if not rotated_vertex[2] == flat_lim:
        print(f'{vertex_to_rotate=}, {rotated_vertex=}, {flat_lim=}')
    assert(rotated_vertex[2] == flat_lim)
    # print(rotated_vertex)
    
    # reconstruct the neighbor
    result = [np.array([101010, 101010, 101010]) for i in range(3)]
    result[vertex_to_rotate_index] = rotated_vertex
    result[axis_index[0]] = axis[0]
    result[axis_index[1]] = axis[1]
    
    # z should be the same for all
    print('result=', result)
    
    assert(t[0][2] == t[1][2] and t[0][2] == t[2][2])
    assert(result[0][2] == result[1][2] and result[0][2] == result[2][2])
    
    return result
    
    
if __name__ == '__main__':
    t  = [np.array([-6,-7,0]), np.array([8,-9,0]), np.array([2,4,0])]
    t1 = [np.array([2,4,0]), np.array([8,-9,0]), np.array([10,8,7])]
    
    result = rotate_to_align(t, t1)
    print(result)