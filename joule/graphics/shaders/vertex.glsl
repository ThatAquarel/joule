#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

uniform mat4 world_transform;
uniform mat4 cam_projection;
uniform mat4 cam_transform;

out vec3 vertex_color;
out vec3 vertex_normal;
out vec3 vertex_frag_pos;

void main() {
    mat4 t = world_transform * cam_transform;
    gl_Position = vec4(position, 1.0) * t * cam_projection;

    vertex_color = color;
    vertex_normal = vec3(vec4(normal, 1.0) * world_transform);
    vertex_frag_pos = vec3(vec4(position, 1.0) * world_transform);
}
