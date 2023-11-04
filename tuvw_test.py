import pygame
import numpy as np
from mobius import *

def get_barycentric_coordinates(x1, y1, x2, y2, x3, y3, x, y):
    T = np.array([
        [x1 - x3, x2 - x3],
        [y1 - y3, y2 - y3]
    ])

    detT = np.linalg.det(T)
    g1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / detT
    g2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / detT
    g3 = 1 - g2 - g1

    return np.array([g1, g2, g3])

def get_mouse_texture_target_barycentric(target_texture_pos, target_texture_size, t_vec, t_tex):
    mouse_pos = pygame.mouse.get_pos()

    barycentric = get_barycentric_coordinates(t_vec[0][0], t_vec[0][1], t_vec[1][0], t_vec[1][1], t_vec[2][0], t_vec[2][1], mouse_pos[0], mouse_pos[1])

    actual_ta = target_texture_pos + np.array([target_texture_size[0] * t_tex[0][0], target_texture_size[1] * t_tex[0][1]])
    actual_tb = target_texture_pos + np.array([target_texture_size[0] * t_tex[1][0], target_texture_size[1] * t_tex[1][1]])
    actual_tc = target_texture_pos + np.array([target_texture_size[0] * t_tex[2][0], target_texture_size[1] * t_tex[2][1]])

    target_pos = actual_ta * barycentric[0] + actual_tb * barycentric[1] + actual_tc * barycentric[2]
    return target_pos

def get_mouse_texture_target_mobius_transform_test(target_texture_pos, target_texture_size, t_vec, t_tex):
    mouse_pos = pygame.mouse.get_pos()

    M_t = findMobiusTransform(t_vec[0], t_vec[1], t_vec[2], t_tex[0], t_tex[1], t_tex[2])
    
    z = mouse_pos[0] + mouse_pos[1] * 1j
    transformed = transform(z, M_t)

    pos = np.array([target_texture_pos[0] + target_texture_size[0] * np.real(transformed), target_texture_pos[1] + target_texture_size[1] * np.imag(transformed)])
    return pos

def get_mouse_texture_target_mobius(target_texture_pos, target_texture_size, t_vec, t_tex, u_vec, u_tex, v_vec, v_tex, w_vec, w_tex):
    mouse_pos = pygame.mouse.get_pos()

    M_t = findMobiusTransform(t_vec[0], t_vec[1], t_vec[2], t_tex[0], t_tex[1], t_tex[2])
    M_u = findMobiusTransform(u_vec[0], u_vec[1], u_vec[2], u_tex[0], u_tex[1], u_tex[2])
    M_v = findMobiusTransform(v_vec[0], v_vec[1], v_vec[2], v_tex[0], v_tex[1], v_tex[2])
    M_w = findMobiusTransform(w_vec[0], w_vec[1], w_vec[2], w_tex[0], w_tex[1], w_tex[2])

    edge_i = t_vec[2][0] + t_vec[2][1] * 1j
    edge_j = t_vec[1][0] + t_vec[1][1] * 1j
    edge_k = t_vec[0][0] + t_vec[0][1] * 1j

    z = mouse_pos[0] + mouse_pos[1] * 1j

    mat_f = log_ratio_interpolator_primary(edge_i, edge_j, edge_k, M_t, M_u, M_v, M_w, z)

    transformed = transform(z, mat_f)

    pos = np.array([target_texture_pos[0] + target_texture_size[0] * np.real(transformed), target_texture_pos[1] + target_texture_size[1] * np.imag(transformed)])
    return pos

class Point:
    _selected = None
    _dragged = None
    _win = None
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z])
    def step(self):
        mouse_pos = np.array(pygame.mouse.get_pos())
        dist = np.sqrt((self.pos[0] - mouse_pos[0]) ** 2 + (self.pos[1] - mouse_pos[1]) ** 2)
        if dist < 10:
            Point._selected = self

    def draw(self):
        color = (255,0,0)
        if self is Point._selected:
            color = (255,255,0)
        pygame.draw.circle(point._win, color, (self.pos[0], self.pos[1]), 5)
        pygame.draw.circle(point._win, color, (self.pos[0], self.pos[1]), 10, 1)

if __name__ == "__main__":
    pygame.init()

    win_dimensions = np.array([1280, 720])
    win = pygame.display.set_mode(win_dimensions)

    texture_surf_org = pygame.image.load('texCheck.png')
    texture_surf = pygame.transform.scale(texture_surf_org, (500,500))

    texture_coords = np.array([[0,0], texture_surf.get_size()])

    offset = np.array([-250, 0, 0])
    target_texture_pos = np.array([650, 100])
    target_texture_size = np.array([500, 500])

    Point._win = win
    points = [
        Point(715, 400, 0),#0
        Point(625, 255, 0),#1
        Point(535, 400, 0),#2
        Point(450, 255, 0),#3
        Point(795, 255, 0),#4
        Point(625, 555, 0),#5
    ]

    for point in points:
        point.pos += offset

    t_vec = np.array([
        0,
        1,
        2,
    ])
    t_tex = np.array([
        [0.7215, 0.555],
        [0.5, 0.1695],
        [0.278, 0.555],
    ])

    u_vec = np.array([
        1,
        3,
        2,
    ])
    u_tex = np.array([
        [0.5, 0.1695],
        [0.0525, 0.1695],
        [0.278, 0.555],
    ])

    v_vec = np.array([
        4,
        1,
        0
    ])
    v_tex = np.array([
        [0.9465, 0.1695],
        [0.5, 0.1695],
        [0.7215, 0.555],
    ])

    w_vec = np.array([
        0,
        2,
        5
    ])
    w_tex = np.array([
        [0.7215, 0.555],
        [0.278, 0.555],
        [0.5, 0.94],
    ])

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if Point._selected:
                    Point._dragged = Point._selected
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                Point._dragged = None
            if event.type == pygame.MOUSEMOTION:
                if Point._dragged:
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    Point._dragged.pos[0] = mouse_pos[0]
                    Point._dragged.pos[1] = mouse_pos[1]
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    pass
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            run = False
        
        # step
        Point._selected = None
        for point in points:
            point.step()

        # mobius_transform = findMobiusTransform(points[t_vec[0]], points[t_vec[1]], points[t_vec[1]], t_tex[0], t_tex[1], t_tex[1])
        mouse_pos = np.array(pygame.mouse.get_pos())
        # z = to_complex(mouse_pos)
        # z_t = transform(z, mobius_transform)
        # print(z_t)

        # z_vec = complex_to_vec(z_t) * texture_coords[1]
        # print(z_vec)
        
        barycentric_coords = get_mouse_texture_target_barycentric(target_texture_pos, target_texture_size, [points[i].pos for i in t_vec], t_tex)
        mobius_test_coords = get_mouse_texture_target_mobius_transform_test(target_texture_pos, target_texture_size, [points[i].pos for i in t_vec], t_tex)
        mobius_coords = get_mouse_texture_target_mobius(target_texture_pos, target_texture_size, [points[i].pos for i in t_vec], t_tex, [points[i].pos for i in u_vec], u_tex, [points[i].pos for i in v_vec], v_tex, [points[i].pos for i in w_vec], w_tex)

        # draw
        win.fill((255,255,255))

        
        win.blit(texture_surf, target_texture_pos)

        pygame.draw.polygon(win, (0,0,0), [(points[i].pos[0], points[i].pos[1]) for i in t_vec], 1)
        pygame.draw.polygon(win, (0,0,0), [(points[i].pos[0], points[i].pos[1]) for i in u_vec], 1)
        pygame.draw.polygon(win, (0,0,0), [(points[i].pos[0], points[i].pos[1]) for i in v_vec], 1)
        pygame.draw.polygon(win, (0,0,0), [(points[i].pos[0], points[i].pos[1]) for i in w_vec], 1)

        texture_triangle_t = [(target_texture_pos[0] + i[0] * texture_coords[1][0], target_texture_pos[1] + i[1] * texture_coords[1][1]) for i in t_tex]
        pygame.draw.polygon(win, (0,0,0), texture_triangle_t, 1)

        # pygame.draw.circle(win, (255,0,0), mouse_pos, 5)
        # pygame.draw.circle(win, (255,0,0), z_vec, 5)
        pygame.draw.circle(win, (255,0,0), barycentric_coords, 5)
        pygame.draw.circle(win, (255,255,0), mobius_test_coords, 5)
        pygame.draw.circle(win, (0,255,255), mobius_coords, 5)

        for point in points:
            point.draw()

        pygame.display.update()

    pygame.quit()