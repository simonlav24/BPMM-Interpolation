
import numpy as np

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
        
    def load_obj(self, path, normalize_texture=False):
        max_vt_x = 0.0
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
            new_vertex_textures = []
            for vt in self.vertices_texture:
                new_vt = (vt[0] / max_vt_x, vt[1] / max_vt_x)
                new_vertex_textures.append(new_vt)
                print(new_vt)
            self.vertices_texture = new_vertex_textures
        
        # create for drawing arrays
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
            # print(len(_vertices))
            
            _vertices_texture.append(vt1)
            _vertices_texture.append(vt2)
            _vertices_texture.append(vt3)
            

        self.vertices_for_drawing = np.array(_vertices)
        self.vertices_texture_for_drawing = np.array(_vertices_texture)
        
        
        

if __name__ == '__main__':
    import pygame
    from random import randint
    pygame.init()

    obj = Model()
    obj.load_obj('./wolf_head.obj')

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
