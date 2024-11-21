from OpenGL.GL import *


def create_vbo(data, v_ptr=3, c_ptr=3, n_ptr=3, return_vbo=False, store_normals=False):
    len_ptr = v_ptr + c_ptr + (n_ptr if store_normals else 0)
    stride = len_ptr * data.itemsize

    # vertex buffer object
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    c_offset = v_ptr * data.itemsize
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(c_offset))
    glEnableVertexAttribArray(1)

    if store_normals:
        n_offset = (v_ptr + c_ptr) * data.itemsize
        glVertexAttribPointer(
            2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(n_offset)
        )
        glEnableVertexAttribArray(2)

    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    if return_vbo:
        return vao, vbo

    return vao


def update_vbo(vbo, data):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def draw_vbo(vao, draw_type, n):
    glBindVertexArray(vao)
    glDrawArrays(draw_type, 0, n)
