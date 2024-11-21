#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 cam_projection;
uniform mat4 cam_position;

// out vec3 originalPosition;
out vec3 v_color;

void main() {
    mat4 rh = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1., 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    gl_Position = vec4(position, 1.0) * rh * cam_position * cam_projection;

    v_color = color;
    // originalPosition = position;
}
