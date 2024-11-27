#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

uniform mat4 cam_projection;
uniform mat4 cam_position;

uniform vec4 light_pos_u;
out vec3 light_pos;

// out vec3 originalPosition;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    mat4 rh = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    gl_Position = vec4(position, 1.0) * rh * cam_position * cam_projection;

    v_color = color;
    v_normal = vec3(vec4(normal, 1.0) * rh);
    v_frag_pos = vec3(vec4(position, 1.0) * rh);

    light_pos = vec3(light_pos_u * inverse(cam_position));
    // originalPosition = position;
}
