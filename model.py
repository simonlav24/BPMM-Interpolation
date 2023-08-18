import numpy as np


class Model3D:
    def __init__(self):
        self.vertices = []
        self.tex_cords = []
        self.faces = []
        self.tex_faces = []
    
    def get_neighbors_of_face(self, face_index):
        main_face = self.faces[face_index]

        neighbors = []

        for face in self.faces:
            count = 0
            if face[0] in main_face:
                count += 1
            if face[1] in main_face:
                count += 1
            if face[2] in main_face:
                count += 1
            if count == 2:
                neighbors.append(face)

        return neighbors



    def create_default_cube(self):
        _vertices = [
            ( 1.000000, -1.000000, -1.000000),
            ( 1.000000, -1.000000,  1.000000),
            (-1.000000, -1.000000,  1.000000),
            (-1.000000, -1.000000, -1.000000),
            ( 1.000000,  1.000000, -0.999999),
            ( 0.999999,  1.000000,  1.000001),
            (-1.000000,  1.000000,  1.000000),
            (-1.000000,  1.000000, -1.000000),
        ]

        _texcoords = [
            (0.250043, 0.749957),
            (0.250043, 0.500000),
            (0.500000, 0.500000),
            (0.500000, 0.250043),
            (0.250043, 0.250043),
            (0.250044, 0.000087),
            (0.500000, 0.999913),
            (0.250043, 0.999913),
            (0.000087, 0.749956),
            (0.000087, 0.500000),
            (0.500000, 0.749957),
            (0.749957, 0.500000),
            (0.500000, 0.000087),
            (0.749957, 0.749957),
        ]
        self.faces = [
            (1, 2, 3),
            (7, 6, 5),
            (4, 5, 1),
            (5, 6, 2),
            (2, 6, 7),
            (0, 3, 7),
            (0, 1, 3),
            (4, 7, 5),
            (0, 4, 1),
            (1, 5, 2),
            (3, 2, 7),
            (4, 0, 7),
        ]

        _texture_triangles = [
            ( 0,  1,  2),
            ( 3,  4,  5),
            ( 6,  7,  0),
            ( 8,  9,  1),
            ( 1,  4,  3),
            (10,  2, 11),
            (10,  0,  2),
            (12,  3,  5),
            (10,  6,  0),
            ( 0,  8,  1),
            ( 2,  1,  3),
            (13, 10, 11),
        ]

        self.vertices = np.array([_vertices[index] for indices in self.faces for index in indices])

        self.tex_faces = np.array([_texcoords[index] for indices in _texture_triangles for index in indices])

def project_3d_to_2d(a, b, c):
    # Calculate the normal vector of the triangle
    normal = np.cross(b - a, c - a)
    
    # Normalize the normal vector
    normal_normalized = normal / np.linalg.norm(normal)
    
    # Project each vertex onto the 2D plane
    a_2d = a - np.dot(a, normal_normalized) * normal_normalized
    b_2d = b - np.dot(b, normal_normalized) * normal_normalized
    c_2d = c - np.dot(c, normal_normalized) * normal_normalized
    
    return a_2d[:2], b_2d[:2], c_2d[:2]

if __name__ == "__main__":
    a = np.array([1, 4, 6])
    b = np.array([3, 5, 2])
    c = np.array([1, 1, 3])

    print(project_3d_to_2d(a, b, c))

    model = Model3D()
    model.create_default_cube()

    n = model.get_neighbors_of_face(0)
    print(f'for face {model.faces[0]} neighbors are: {n}')