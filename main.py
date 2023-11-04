
from OpenGL.GL import *
import numpy as np
import transformations
import pygame
from model import *
from PIL import Image
from objLoader import *

def load_program(vertex_source, fragment_source):
    """ load shaders and create program. return program opengl index """
    vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source)
    if vertex_shader == 0:
        return 0

    fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source)
    if fragment_shader == 0:
        return 0

    program = glCreateProgram()

    if program == 0:
        return 0

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)

    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
        glDeleteProgram(program)
        return 0

    return program

def load_shader(shader_type, source):
    """ load and compile a single shader. return shader opengl index """
    shader = glCreateShader(shader_type)

    if shader == 0:
        return 0

    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS, None) == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        print(info_log)
        glDeleteProgram(shader)
        return 0

    return shader

def load_texture(filename):
    """ load image and create texture. returns texture """
    img = Image.open(filename, 'r').convert("RGB")
    img_data = np.array(img, dtype=np.uint8)
    w, h = img.size

    texture = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    return texture

def preview(model_path, texture_path):
    model = Model()
    model.load_obj(model_path)

    with open(r'./Shaders/vertex_shader.glsl', 'r') as v_shader:
        vertex_shader_source = v_shader.read()

    with open(r'./Shaders/fragment_shader.glsl', 'r') as f_shader:
        fragment_shader_source = f_shader.read()

    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.DOUBLEBUF|pygame.OPENGL|pygame.HWSURFACE)

    glViewport(0, 0, width, height) # creates the viewport. instead of -1 to 1 it make 0 to width
    
    projection_matrix = transformations.perspective(45, width/height, 0.1, 500) # create perspective matrix
    
    model_matrix = np.identity(4, dtype=np.float32) # model matrix to transform in model space
    
    view_matrix = np.identity(4, dtype=np.float32)
    view_z = -300
    view_matrix[-1, :-1] = (0, 0, view_z) # modifies the last row of the view_matrix to set its values
    """The indexing [-1, :-1] means that it's referring to the last row of the matrix (-1), but excluding the last element (:-1), which is used for perspective transformations and is typically set to 0.
    The tuple (0, 0, -10) is being assigned to this row, which suggests that the transformation is shifting the entire scene along the negative z-axis by a distance of 10 units."""

    program = load_program(vertex_shader_source, fragment_shader_source) # create the program

    uMVMatrix = glGetUniformLocation(program, "uMVMatrix") # get the location of uMVMatrix in the program (uniform variable)
    uPMatrix = glGetUniformLocation(program, "uPMatrix") # get the location of uPMatrix in the program (uniform variable)
    sTexture = glGetUniformLocation(program, "sTexture")  # get the location of sTexture in the program (uniform variable)
       
    aVertex = glGetAttribLocation(program, "aVertex") # per vertex data, create (and get the location of) the attrib aVertex
    aTexCoord = glGetAttribLocation(program, "aTexCoord") # per vertex data, create (and get the location of) the attrib aTexCoord

    glUseProgram(program) # select the program
    glEnableVertexAttribArray(aVertex) # enable the attrib for rendering
    glEnableVertexAttribArray(aTexCoord) # enable the attrib for rendering

    texture = load_texture(texture_path)

    glActiveTexture(GL_TEXTURE0) # select the texture unit GL_TEXTURE0 (the first texture unit)
    glBindTexture(GL_TEXTURE_2D, texture) # bind it to texture we just loaded
    glUniform1i(sTexture, 0) # the uniform var sTexture is now linked to the texture (1 integer)

    glEnable(GL_DEPTH_TEST)

    run = True
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEMOTION:
                x, y = event.rel
                if any(event.buttons):
                    model_matrix = model_matrix.dot(transformations.rotate(y, -1, 0, 0)).dot(transformations.rotate(x, 0, -1, 0))
            if event.type == pygame.MOUSEWHEEL:
                    view_z += event.y * 10
                    view_matrix[-1, :-1] = (0, 0, view_z)
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            run = False
        
        # DRAW
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # clear the color and depth buffer

        # draw model
        glVertexAttribPointer(aVertex, 3, GL_FLOAT, GL_FALSE, 0, model.vertices_for_drawing) # link aVertex attribute to model vertices
        glVertexAttribPointer(aTexCoord, 2, GL_FLOAT, GL_FALSE, 0, model.vertices_texture_for_drawing) # link aTexCoord attribute to model tex coords

        mv_matrix = np.dot(model_matrix, view_matrix) # create modelview matrix
        # link shaders uniforms
        glUniformMatrix4fv(uMVMatrix, 1, GL_FALSE, mv_matrix) # the uniform var uMVMatrix is now linked to mv_matrix
        glUniformMatrix4fv(uPMatrix, 1, GL_FALSE, projection_matrix) # the uniform var uPMatrix is now linked to projection_matrix

        glDrawArrays(GL_TRIANGLES, 0, len(model.vertices_for_drawing)) # draw 

        pygame.display.flip()
    pygame.quit()

        