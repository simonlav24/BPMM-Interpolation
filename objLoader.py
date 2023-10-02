import numpy as np
from rotatetoalign import *
from mobius import *
from desmos_tools import *
from drawer_helper import Drawer

DESMOS_PRINT = True

class Model:
    def __init__(self):
        self.vertices = [] # at index (x, y, z)
        self.vertices_texture = [] # at index (u, v)
        self.faces = [] # at index {'v': vertex_index, 'vn': vertex_normal_index, 'vt': vertex_texture_index}
        self.neighbors = [] # at index (n1_face_index, n2_face_index, n3_face_index)
        
        self.vertices_for_drawing = [] # the vertices to actual draw f1v1, f1v2, f1v3, f2v1, f2v2, ...
        self.vertices_texture_for_drawing = [] # same but vt
        
    def get_face_vertices(self, face_index):
        face = self.faces[face_index]
        vertices = [self.vertices[face[i]['v']] for i in range(3)]
        return vertices
        
    def get_face_vertices_indices(self, face_index):
        face = self.faces[face_index]
        vertices = [face[i]['v'] for i in range(3)]
        return vertices
    
    def get_neighbors(self, face_index):
        return self.neighbors[face_index]
    
    def save_obj(self, path):
        output = '\n'
        for v in self.vertices:
            output += f'v {v[0]} {v[1]} {v[2]}\n'

        output += '\n\n'
    
        for vt in self.vertices_texture:
            output += f'vt {vt[0]} {vt[1]}\n'

        output += '\n\n'

        for face in self.faces:
            a = face[0]
            b = face[1]
            c = face[2]

            output += f"f {a['v']+1}/{a['vt']+1} {b['v']+1}/{b['vt']+1} {c['v']+1}/{c['vt']+1} \n"

        output += '\n'

        with open(path, 'w+') as file:
            file.write(output)
        
    def load_obj(self, path, normalize_texture=False):
        max_vt_x = 0.0 # normalizers for textures that > 1.0
        max_vt_y = 0.0
        with open(path, 'r') as file:
            for line in file.readlines():
                if line.startswith('vn'):
                    pass
                if line.startswith('vt'):
                    x, y = line[2:].split()
                    max_vt_x = max(max_vt_x, float(x))
                    max_vt_y = max(max_vt_y, float(y))
                    self.vertices_texture.append((float(x), float(y)))
                elif line.startswith('v'):
                    x, y, z = line[1:].split()
                    self.vertices.append((float(x), float(y), float(z)))
                elif line.startswith('f'):
                    abc = line[1:].split()
                    new_face = []
                    for element in abc:
                        splitted = element.split('/')
                        if len(splitted) == 2:
                            new_face.append({'v': int(splitted[0]) - 1, 'vt': int(splitted[1]) - 1})
                        elif len(splitted) == 3:
                            new_face.append({'v': int(splitted[0]) - 1, 'vt': int(splitted[1]) - 1 if splitted[1] != '' else 0, 'vn': int(splitted[2]) - 1})
                        else:
                            new_face.append({'v': int(splitted) - 1, 'vt': 0, 'vn': 0})
                    self.faces.append(new_face)
                    
        # create neighbors
        face_indices = [(face[0]['v'], face[1]['v'], face[2]['v']) for face in self.faces]
        
        for i, face_source in enumerate(face_indices):
            face_neighbors = []
            for j, face in enumerate(face_indices):
                count = 0
                
                for k in range(3):
                    if face_source[k] in face:
                        count += 1
                
                if count == 2:
                    face_neighbors.append(j)
            self.neighbors.append(face_neighbors)
        
        if normalize_texture:
            # normalize textures when vt > 1.0
            new_vertex_textures = []
            for vt in self.vertices_texture:
                new_vt = (vt[0] / max_vt_x, vt[1] / max_vt_x)
                new_vertex_textures.append(new_vt)
            self.vertices_texture = new_vertex_textures

            self.save_obj('output.obj')
        
        # create for drawing arrays -> will change to new taignles and new vertices
        _vertices = []
        _vertices_texture = []
        for face in self.faces:
            v1 = self.vertices[face[0]['v']]
            v2 = self.vertices[face[1]['v']]
            v3 = self.vertices[face[2]['v']]
            
            vt1 = self.vertices_texture[face[0]['vt']]
            vt2 = self.vertices_texture[face[1]['vt']]
            vt3 = self.vertices_texture[face[2]['vt']]
            
            _vertices.append(v1)
            _vertices.append(v2)
            _vertices.append(v3)
            
            _vertices_texture.append(vt1)
            _vertices_texture.append(vt2)
            _vertices_texture.append(vt3)
            

        self.vertices_for_drawing = np.array(_vertices)
        self.vertices_texture_for_drawing = np.array(_vertices_texture)
        
        ##########################################################
        #                      Mobius Area
        ##########################################################

        divided_model = Model()
        divide_factor = 5

        vertex_index = 0
        face_index = 0

        # try to find mobiuses
        for face_index, face in enumerate(self.faces):
            '''
            face is: [{'v': index, 'vt': index}, {'v': index, 'vt': index}, {'v': index, 'vt': index}]
            '''

            # print(f'face {face_index}: {face}')
            # print(self.get_neighbors(face_index))
            
            # calculcate normal of the face
            vertices = np.array([np.array(self.vertices[face[i]['v']]) for i in range(3)])
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            # print(vertices)
            # print(normal)
            
            n = normal
            z = np.array([0, 0, 1])
            v = np.cross(n, z)
            # print(f'{v=}')
            s = np.linalg.norm(v)
            c = np.dot(n, z)
            
            I = np.identity(3)
            V_x = np.array([[0    , -v[2],  v[1]],
                            [ v[2], 0    , -v[0]],
                            [-v[1],  v[0], 0    ]])
            
            last_part = (1 - c) / (s * s)
            # print(last_part)
            
            ##########################################################
            # 1. rotate the triangle so that it will be flat on z
            ##########################################################

            rotation_matrix = I + V_x + np.dot(V_x, V_x) * last_part
            # print(rotation_matrix)

            middle_rotated = np.array([np.dot(rotation_matrix, i) for i in vertices])
            # print(middle_rotated)
            # this is the final middle triangle we can omit the z.
            
            # neighbors:
            neighbors_faces_indices = self.get_neighbors(face_index)
            neighbors_vertices = [np.array([self.vertices[j['v']] for j in self.faces[i]]) for i in neighbors_faces_indices]
            if len(neighbors_vertices) != 3:
                continue
            # neighbors_vertices are list of 3 np.array with their actual vertices
            
            ##########################################################
            # 2. rotate the neighbors to be in the same area as the middle. not yet rotated to be aligned on z
            ##########################################################

            # rotate the neighbors with the same rotation used for middle triangle
            rotated_neighbors = []
            for neighbor in neighbors_vertices:
                rotated_neighbor = np.array([np.dot(rotation_matrix, i) for i in neighbor])
                rotated_neighbors.append(rotated_neighbor)

            if DESMOS_PRINT and False:
                print('<current neighbors>')
                print(polygon_to_desmos(middle_rotated))
                for n in rotated_neighbors:
                    print(polygon_to_desmos(n))
                print('</current neighbors>')
            

            ##########################################################
            # 3. rotate the neighbors to be aligned on z plane (verified in desmos)
            ##########################################################

            # rotate every neighbor along connecting edge
            # print('neighbors before:')
            # print(rotated_neighbors)
            
            # Drawer(triangles_to_draw=rotated_neighbors, file='1')

            rotated_flat_neighbors = []
            for i, neighbor in enumerate(rotated_neighbors):
                # print(f'Iteration {i}')
                neighbor_along = rotate_to_align(middle_rotated, neighbor)
                rotated_flat_neighbors.append(neighbor_along)

            # Drawer(triangles_to_draw=rotated_flat_neighbors, file='2')

            if DESMOS_PRINT and False:
                print('<aligned neighbors>')
                print(polygon_to_desmos_2d(middle_rotated))
                for n in rotated_flat_neighbors:
                    print(polygon_to_desmos_2d(n))
                print('</aligned neighbors>')

            ##########################################################
            # 4. get everything ready for mobius calculations
            ##########################################################

            neighbors_textures = [np.array([self.vertices_texture[j['v']] for j in self.faces[i]]) for i in neighbors_faces_indices]

            triangle_t_vec_2d = middle_rotated
            triangle_t_tex_2d = np.array([np.array(self.vertices_texture[face[i]['v']]) for i in range(3)])

            triangle_u_vec_2d = rotated_flat_neighbors[0]
            triangle_u_tex_2d = neighbors_textures[0]

            triangle_v_vec_2d = rotated_flat_neighbors[1]
            triangle_v_tex_2d = neighbors_textures[1]

            triangle_w_vec_2d = rotated_flat_neighbors[2]
            triangle_w_tex_2d = neighbors_textures[2]

            M_t = findMobiusTransform(triangle_t_vec_2d[0], triangle_t_vec_2d[1], triangle_t_vec_2d[2], triangle_t_tex_2d[0], triangle_t_tex_2d[1], triangle_t_tex_2d[2])
            M_u = findMobiusTransform(triangle_u_vec_2d[0], triangle_u_vec_2d[1], triangle_u_vec_2d[2], triangle_u_tex_2d[0], triangle_u_tex_2d[1], triangle_u_tex_2d[2])
            M_v = findMobiusTransform(triangle_v_vec_2d[0], triangle_v_vec_2d[1], triangle_v_vec_2d[2], triangle_v_tex_2d[0], triangle_v_tex_2d[1], triangle_v_tex_2d[2])
            M_w = findMobiusTransform(triangle_w_vec_2d[0], triangle_w_vec_2d[1], triangle_w_vec_2d[2], triangle_w_tex_2d[0], triangle_w_tex_2d[1], triangle_w_tex_2d[2])

            # j: the point shared by t, u, v
            edge_j = get_shared_point(triangle_t_vec_2d, triangle_u_vec_2d, triangle_v_vec_2d)
            # Drawer([triangle_t_vec_2d, triangle_u_vec_2d, triangle_v_vec_2d], [edge_j])
            assert edge_j is not None

            # i: the point shared by t, u, w
            edge_i = get_shared_point(triangle_t_vec_2d, triangle_u_vec_2d, triangle_w_vec_2d)
            # Drawer([triangle_t_vec_2d, triangle_u_vec_2d, triangle_v_vec_2d], [edge_j])
            assert edge_i is not None

            # k: the point shared by t, v, w
            edge_k = get_shared_point(triangle_t_vec_2d, triangle_v_vec_2d, triangle_w_vec_2d)
            # Drawer([triangle_t_vec_2d, triangle_u_vec_2d, triangle_v_vec_2d], [edge_j])
            assert edge_k is not None

            j_complex = to_complex(edge_j)
            i_complex = to_complex(edge_i)
            k_complex = to_complex(edge_k)

            ##########################################################
            # 4.5. test
            ##########################################################

            if False:
                # point i on t should return upon interpolation the vt of i
                i_texture_result = get_shared_point(triangle_t_tex_2d, triangle_u_tex_2d, triangle_w_tex_2d)

                test_point = i_complex + j_complex * 0.01 + k_complex * 0.01

                # Drawer(points_to_draw=[edge_i, edge_k, edge_j, (np.real(test_point), np.imag(test_point))])

                result_complex = log_ratio_interpolator_primary_and_transform(i_complex, j_complex, k_complex, M_t, M_u, M_v, M_w, test_point)


            ##########################################################
            # 5. divide and create new tiangles with mobius texture
            ##########################################################

            original_triangle_t = vertices

            # were dividing triangle t into 

            a = original_triangle_t[0]
            b = original_triangle_t[1]
            c = original_triangle_t[2]

            mobius_a = triangle_t_vec_2d[0]
            mobius_b = triangle_t_vec_2d[1]
            mobius_c = triangle_t_vec_2d[2]

            step_ab = (b - a) / divide_factor
            step_ac = (c - a) / divide_factor
            step_bc = (c - b) / divide_factor

            mobius_step_ab = (mobius_b - mobius_a) / divide_factor
            mobius_step_ac = (mobius_c - mobius_a) / divide_factor
            mobius_step_bc = (mobius_c - mobius_b) / divide_factor

            for column_index in range(divide_factor):
                triangles_in_columns = (column_index * 2) + 1
                for row_index in range(triangles_in_columns):
                    if row_index % 2 == 0:
                        # right triangle
                        new_a = a + column_index * step_ab
                        mobius_new_a = mobius_a + column_index * mobius_step_ab
                        new_a = new_a + (row_index // 2) * step_bc
                        mobius_new_a = mobius_new_a + (row_index // 2) * mobius_step_bc

                        new_b = new_a + step_ab
                        mobius_new_b = mobius_new_a + mobius_step_ab
                        new_c = new_a + step_ac
                        mobius_new_c = mobius_new_a + mobius_step_ac
                    else:
                        # inverted triangle
                        new_a = a + column_index * step_ab
                        mobius_new_a = mobius_a + column_index * mobius_step_ab
                        new_a = new_a + ((row_index - 1) // 2) * step_bc
                        mobius_new_a = mobius_new_a + ((row_index - 1) // 2) * mobius_step_bc

                        new_b = new_a + step_ac
                        mobius_new_b = mobius_new_a + mobius_step_ac
                        new_c = new_a + step_bc
                        mobius_new_c = mobius_new_a + mobius_step_bc
                    
                    vt_a = log_ratio_interpolator_primary_and_transform(i_complex, j_complex, k_complex, M_t, M_u, M_v, M_w, to_complex(mobius_new_a))
                    vt_b = log_ratio_interpolator_primary_and_transform(i_complex, j_complex, k_complex, M_t, M_u, M_v, M_w, to_complex(mobius_new_b))
                    vt_c = log_ratio_interpolator_primary_and_transform(i_complex, j_complex, k_complex, M_t, M_u, M_v, M_w, to_complex(mobius_new_c))

                    vt_a = complex_to_vec(vt_a)
                    if np.isnan(vt_a[0]): vt_a[0] = 0
                    if np.isnan(vt_a[1]): vt_a[1] = 0
                    vt_b = complex_to_vec(vt_b)
                    if np.isnan(vt_b[0]): vt_b[0] = 0
                    if np.isnan(vt_b[1]): vt_b[1] = 0
                    vt_c = complex_to_vec(vt_c)
                    if np.isnan(vt_c[0]): vt_c[0] = 0
                    if np.isnan(vt_c[1]): vt_c[1] = 0

                    divided_model.vertices.append(new_a)
                    divided_model.vertices_texture.append(vt_a)
                    f0 = {'v': vertex_index, 'vt': vertex_index}
                    vertex_index += 1
                    divided_model.vertices.append(new_b)
                    divided_model.vertices_texture.append(vt_b)
                    f1 = {'v': vertex_index, 'vt': vertex_index}
                    vertex_index += 1
                    divided_model.vertices.append(new_c)
                    divided_model.vertices_texture.append(vt_c)
                    f2 = {'v': vertex_index, 'vt': vertex_index}
                    vertex_index += 1
                    divided_model.faces.append((f0, f1, f2))
        


        # todo: remove this redundant vt
        divided_model.vertices_texture.append((0,0))
        divided_model.save_obj(f'divided_{divide_factor}.obj')
            




        check_point = []

if __name__ == '__main__':
    import pygame
    from random import randint
    pygame.init()

    obj = Model()
    obj.load_obj('./wolf_head_fixed.obj')

    # for face in obj.faces:
        # print(face)
        # if len(face) != 3:
            # print('--------------------------------------')
     
    lll = [(obj.vertices[face[0]['v']], obj.vertices[face[1]['v']], obj.vertices[face[2]['v']]) for face in obj.faces]
    # print(lll)
    
    winWidth = 1280
    winHeight = 720
    win = pygame.display.set_mode((winWidth,winHeight))

    fps_clock = pygame.time.Clock()
    
    offset_x = 500
    offset_y = 500
    scale = 5
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            run = False
        
        win.fill((0,0,0))
        
        # draw model
        for face in lll:
            vertices = []
            for vertex in face:
                x = vertex[0] * scale + offset_x
                y = vertex[1] * scale + offset_y
                z = vertex[2] * scale
                vertices.append((x, y))
            
            pygame.draw.polygon(win, (255, 255, 255), vertices, 1)
        
        # draw neighbors
        num_of_faces = len(obj.faces)
        random_face = randint(0, num_of_faces - 1)
        
        neighbors = obj.neighbors[random_face]
        print(neighbors)
        for n in neighbors:
            neighbor_vertices = obj.get_face_vertices(n)
            neighbor_vertices = [(i[0] * scale + offset_x, i[1] * scale + offset_y) for i in neighbor_vertices]
            pygame.draw.polygon(win, (255, 255, 0), neighbor_vertices, 1)
            
        random_face_vertices = obj.get_face_vertices(random_face)
        random_face_vertices = [(i[0] * scale + offset_x, i[1] * scale + offset_y) for i in random_face_vertices]
        pygame.draw.polygon(win, (255, 0, 0), random_face_vertices, 1)
        
        pygame.display.update()
        fps_clock.tick(2)
