from OpenGL.GL import *


def create_vbo(data, v_ptr=3, c_ptr=3, return_vbo=False):
    stride = (v_ptr + c_ptr) * data.itemsize

    # vertex buffer object
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    offset = v_ptr * data.itemsize
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    if return_vbo:
        return vao, vbo

    return vao


def update_vbo(vbo, data):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


# def draw_vbo(vbo, stride, draw_type, n, v_ptr=3, c_ptr=3):
#     # bind to VBO
#     glBindBuffer(GL_ARRAY_BUFFER, vbo)

#     # enable vertex followed by color within VBOs
#     glEnableClientState(GL_VERTEX_ARRAY)
#     glVertexPointer(v_ptr, GL_FLOAT, stride, ctypes.c_void_p(0))
#     glEnableClientState(GL_COLOR_ARRAY)

#     # calculate color offset (assuming data is tightly packed)
#     # color comes after vertex
#     size = stride // (v_ptr + c_ptr)
#     glColorPointer(c_ptr, GL_FLOAT, stride, ctypes.c_void_p(v_ptr * size))

#     # draw VBO
#     glDrawArrays(draw_type, 0, n)

#     glDisableClientState(GL_VERTEX_ARRAY)
#     glDisableClientState(GL_COLOR_ARRAY)
#     glBindBuffer(GL_ARRAY_BUFFER, 0)


def draw_vbo(vao, draw_type, n):
    glBindVertexArray(vao)
    glDrawArrays(draw_type, 0, n)
