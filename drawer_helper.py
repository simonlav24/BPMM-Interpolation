import pygame
from random import randint

def random_color():
    return (randint(100, 220), randint(100, 220), randint(100, 220))

def three_decimals(f):
    return "{:.3f}".format(f)

class Drawer:
    def __init__(self, triangles_to_draw = [], points_to_draw = [], edges_to_draw = [], file='test'):
        pygame.font.init()
        # draw them all
        if len(triangles_to_draw) > 0:
            max_x = triangles_to_draw[0][0][0]
            min_x = triangles_to_draw[0][0][0]
            max_y = triangles_to_draw[0][0][1]
            min_y = triangles_to_draw[0][0][1]

        if len(points_to_draw) > 0:
            max_x = points_to_draw[0][0]
            min_x = points_to_draw[0][0]
            max_y = points_to_draw[0][1]
            min_y = points_to_draw[0][1]

        if len(edges_to_draw) > 0:
            max_x = edges_to_draw[0][0][0]
            min_x = edges_to_draw[0][0][0]
            max_y = edges_to_draw[0][0][1]
            min_y = edges_to_draw[0][0][1]

        for triangle in triangles_to_draw:
            for vertex in triangle:
                max_x = max(max_x, vertex[0])
                min_x = min(min_x, vertex[0])
                max_y = max(max_y, vertex[1])
                min_y = min(min_y, vertex[1])
        
        for point in points_to_draw:
            max_x = max(max_x, point[0])
            min_x = min(min_x, point[0])
            max_y = max(max_y, point[1])
            min_y = min(min_y, point[1])

        for edge in edges_to_draw:
            for vertex in edge:
                max_x = max(max_x, vertex[0])
                min_x = min(min_x, vertex[0])
                max_y = max(max_y, vertex[1])
                min_y = min(min_y, vertex[1])

        self.extra_space = 100

        self.new_origin = (min_x, min_y)
        self.scale_factor = 1280 / (max_x - min_x)

        width = (max_x - min_x) * self.scale_factor + self.extra_space // 2
        height = (max_y - min_y) * self.scale_factor + self.extra_space // 2
        self.width = width
        self.height = height
        self.new_origin = (min_x * self.scale_factor, min_y * self.scale_factor)

        font = pygame.font.SysFont("Calibri", 16)

        surf = pygame.Surface((width, height))
        surf.fill((255,255,255))

        for triangle in triangles_to_draw:
            org_points = [point for point in triangle]
            points = [self.p(point) for point in triangle]
            pygame.draw.polygon(surf, random_color(), points, 1)
            for i, p in enumerate(points):
                text = font.render(f'[{i}]({three_decimals(org_points[i][0])}, {three_decimals(org_points[i][1])})', True, (0,0,0))
                surf.blit(text, p)

        for point in points_to_draw:
            transformed = self.p(point)
            x = transformed[0]
            y = transformed[1]
            pygame.draw.circle(surf, (255,0,0), (x,y), 5)
            text = font.render(f'({three_decimals(point[0])}, {three_decimals(point[1])})', True, (0,0,0))
            surf.blit(text, transformed)

        for edge in edges_to_draw:
            start = self.p(edge[0])
            end = self.p(edge[1])
            pygame.draw.line(surf, random_color(), start, end)
            for i, p in enumerate(edge):
                text = font.render(f'({three_decimals(edge[i][0])}, {three_decimals(edge[i][1])})', True, (0,0,0))
                surf.blit(text, self.p(p))

        pygame.image.save(surf, f'./{file}.png')
        print(f'[DRAWER] saved to "./{file}.png"')

    def p(self, point):
        x = point[0] * self.scale_factor - self.new_origin[0] + self.extra_space // 4
        y = point[1] * self.scale_factor - self.new_origin[1] + self.extra_space // 4
        y = self.height - y
        return (x, y)

if __name__ == '__main__':
    Drawer(triangles_to_draw=[((1,3), (3,8), (14, 6))], points_to_draw=[(5, 7)], edges_to_draw=[((1, 5), (10, 3))])
