import numpy as np
from random import randint, uniform
from desmos_tools import *
from drawer_helper import Drawer

DESMOS_PRINT = True

def vertex_equall(v1, v2):
    result = True
    result &= np.isclose(v1[0], v2[0])
    result &= np.isclose(v1[1], v2[1])
    result &= np.isclose(v1[2], v2[2])
    return result

def check_for_edge_intersection(e1, e2):
    # check if about close
    if (
        np.isclose(e1[0][0], e2[0][0]) and np.isclose(e1[0][1], e2[0][1]) and
        np.isclose(e1[1][0], e2[1][0]) and np.isclose(e1[1][1], e2[1][1])
        ):
        return False

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2
    
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and 
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    p1, q1 = e1
    p2, q2 = e2
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if (o1 != o2 and o3 != o4) or (o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, q2, q1)) or (o3 == 0 and on_segment(p2, p1, q2)) or (o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False

def check_for_triangle_intersection(t1, t2):
    ''' not working for triangles that share an edge '''
    edges_1 = [(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])]
    edges_2 = [(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])]

    # check if all vertices are about the same
    # Drawer([t1, t2])
    count = 0
    for point1 in t1:
        for point2 in t2:
            if np.isclose(point1[0], point2[0]) and np.isclose(point1[1], point2[1]):
                count += 1
    # print(f'{count=}')
    if count == 3:
        return True

    for edge_1 in edges_1:
        for edge_2 in edges_2:
            
            # check if one of the points are about the same
            if (np.isclose(edge_1[0][0], edge_2[0][0]) and np.isclose(edge_1[0][1], edge_2[0][1]) or
                np.isclose(edge_1[1][0], edge_2[1][0]) and np.isclose(edge_1[1][1], edge_2[1][1]) or
                np.isclose(edge_1[0][0], edge_2[1][0]) and np.isclose(edge_1[0][1], edge_2[1][1]) or
                np.isclose(edge_1[1][0], edge_2[0][0]) and np.isclose(edge_1[1][1], edge_2[0][1])
                ):
                continue

            if check_for_edge_intersection(edge_1, edge_2):
                return True
    return False

def check_if_point_inside_triangle(point, triangle):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, triangle[0], triangle[1]) < 0.0
    b2 = sign(point, triangle[1], triangle[2]) < 0.0
    b3 = sign(point, triangle[2], triangle[0]) < 0.0

    return ((b1 == b2) and (b2 == b3))

def check_if_any_point_inside_triangle(t1, t2):
    for p in t1:
        if check_if_point_inside_triangle(p, t2):
            return True
    for p in t2:
        if check_if_point_inside_triangle(p, t1):
            return True
    return False

def check_if_triangles_on_different_orientation(triangle, potential_neighbor, odd_point):
    if check_for_triangle_intersection(triangle, potential_neighbor):
        return False
    if check_if_point_inside_triangle(odd_point, triangle):
        return False
    return True

def find_intersection_points(x1, y1, r1, x2, y2, r2):
    """ find intersection point between two circles """
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if d > r1 + r2:
        # Circles do not intersect
        return None
    elif d < np.abs(r1 - r2):
        # One circle is inside the other
        return None
    elif d == 0 and r1 == r2:
        # Circles coincide
        return None
    else:
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d
        x4 = x3 + h * (y2 - y1) / d
        y4 = y3 - h * (x2 - x1) / d
        x5 = x3 - h * (y2 - y1) / d
        y5 = y3 + h * (x2 - x1) / d
        return (x4, y4), (x5, y5)

def rotate_to_align(triangle, neighbor):
    result = []
    z_value = triangle[0][2]
    # find same vertices and edge
    equal_vertices_indices = []
    if vertex_equall(triangle[0], neighbor[0]):
        equal_vertices_indices.append(0)
    if vertex_equall(triangle[1], neighbor[0]):
        equal_vertices_indices.append(0)
    if vertex_equall(triangle[2], neighbor[0]):
        equal_vertices_indices.append(0)

    if vertex_equall(triangle[0], neighbor[1]):
        equal_vertices_indices.append(1)
    if vertex_equall(triangle[1], neighbor[1]):
        equal_vertices_indices.append(1)
    if vertex_equall(triangle[2], neighbor[1]):
        equal_vertices_indices.append(1)

    if vertex_equall(triangle[0], neighbor[2]):
        equal_vertices_indices.append(2)
    if vertex_equall(triangle[1], neighbor[2]):
        equal_vertices_indices.append(2)
    if vertex_equall(triangle[2], neighbor[2]):
        equal_vertices_indices.append(2)

    odd_vertex = [0,1,2]
    for i in equal_vertices_indices:
        odd_vertex.remove(i)
    odd_vertex_index = odd_vertex[0]

    edge_indices = [0,1,2]
    edge_indices.remove(odd_vertex_index)
    edge_vertices = [neighbor[edge_indices[0]], neighbor[edge_indices[1]]]

    line1_indices = [edge_indices[0], odd_vertex_index]
    line2_indices = [edge_indices[1], odd_vertex_index]

    line1_vertices = [neighbor[line1_indices[0]], neighbor[line1_indices[1]]]
    line2_vertices = [neighbor[line2_indices[0]], neighbor[line2_indices[1]]]

    p1, p2 = line1_vertices
    line1_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    p1, p2 = line2_vertices
    line2_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

    x1 = edge_vertices[0][0]
    y1 = edge_vertices[0][1]
    r1 = line1_dist

    x2 = edge_vertices[1][0]
    y2 = edge_vertices[1][1]
    r2 = line2_dist
    intersection_points = find_intersection_points(x1, y1, r1, x2, y2, r2)

    # create the potential neighbors. this method should ensure that the index order is correct
    edge_vertices_out = edge_vertices.copy()
    potential_neighbor1 = []
    for i in range(3):
        if i == odd_vertex_index:
            potential_neighbor1.append(intersection_points[0])
        else:
            potential_neighbor1.append(edge_vertices_out.pop(0))

    edge_vertices_out = edge_vertices.copy()
    potential_neighbor2 = []
    for i in range(3):
        if i == odd_vertex_index:
            potential_neighbor2.append(intersection_points[1])
        else:
            potential_neighbor2.append(edge_vertices_out.pop(0))

    # print(polygon_to_desmos_2d(triangle))
    # print(polygon_to_desmos_2d(potential_neighbor1))
    # print(polygon_to_desmos_2d(potential_neighbor2))

    Drawer([triangle, potential_neighbor1, potential_neighbor2], [intersection_points[0], intersection_points[1]])
    check1 = check_if_triangles_on_different_orientation(triangle, potential_neighbor1, intersection_points[0])
    check2 = check_if_triangles_on_different_orientation(triangle, potential_neighbor2, intersection_points[1])

    assert(check1 != check2)

    if check1:
        return potential_neighbor1
    else:
        return potential_neighbor2



def rotate_to_align_not_working_2(triangle, neighbor):
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

    simple_test = False
    if simple_test:
        t  = [np.array([-6,-7,0]), np.array([8,-9,0]), np.array([2,4,0])]
        t1 = [np.array([2,4,0]), np.array([8,-9,0]), np.array([10,8,7])]
        
        result = rotate_to_align(t, t1)
        print(result)

    test_intersecting_edges = True
    if test_intersecting_edges:
        e1 = ((-4.33, 7.55), (1.68, -0.93))
        e2 = ((-1, -6.65), (5.3, 2.23))
        print(check_for_edge_intersection(e1, e2))
        assert check_for_edge_intersection(e1, e2) == False

        e1 = ((4.67,3.43), (-1.9,-5.68))
        e2 = ((-3,-5), (3.67,4.53))
        print(check_for_edge_intersection(e1, e2))
        assert check_for_edge_intersection(e1, e2) == False

        e1 = ((5.44,8.2), (1.93,-4.4))
        e2 = ((5,-5), (2.8,3.54))
        print(check_for_edge_intersection(e1, e2))
        assert check_for_edge_intersection(e1, e2) == True

        a, b, c, d = uniform(-30,30), uniform(-30,30), uniform(-30,30), uniform(-30,30)
        e1 = ((a, b), (c, d))
        e2 = ((a, b), (c, d))
        print(check_for_edge_intersection(e1, e2))
        assert check_for_edge_intersection(e1, e2) == False

        # test for 'check_if_any_point_inside_triangle' and for 'check_if_point_inside_triangle' for points that are on the triangle
        check_if_any_point_inside_triangle()
