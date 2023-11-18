def vertex_to_desmos(v):
    return f'p\\left({v[0]},{v[1]},{v[2]}\\right)' 

def vertex_to_desmos_custom_z(v, z):
    return f'p\\left({v[0]},{v[1]},{z}\\right)'

def vertex_to_desmos_2d(v):
    return f'\\left({v[0]},{v[1]}\\right)' 

def polygon_to_desmos(p):
    output = f'\\left['

    output += vertex_to_desmos(p[0]) + ','
    output += vertex_to_desmos(p[1]) + ','
    output += vertex_to_desmos(p[2]) + ','
    output += vertex_to_desmos(p[0])

    output += '\\right]'
    return output

def polygon_to_desmos_custom_z(p, z):
    output = f'\\left['

    output += vertex_to_desmos_custom_z(p[0], z) + ','
    output += vertex_to_desmos_custom_z(p[1], z) + ','
    output += vertex_to_desmos_custom_z(p[2], z) + ','
    output += vertex_to_desmos_custom_z(p[0], z)

    output += '\\right]'
    return output

def polygon_to_desmos_2d(p):
    output = f'\\left['

    output += vertex_to_desmos_2d(p[0]) + ','
    output += vertex_to_desmos_2d(p[1]) + ','
    output += vertex_to_desmos_2d(p[2]) + ','
    output += vertex_to_desmos_2d(p[0])

    output += '\\right]'
    return output

def edge_to_desmos_2d(p):
    output = f'\\left['
    output += vertex_to_desmos_2d(p[0]) + ','
    output += vertex_to_desmos_2d(p[1])
    output += '\\right]'
    return output