#version 330 core

uniform vec3 light_pos;

in vec3 v_color;
in vec3 v_normal;
in vec3 v_frag_pos;

out vec4 fragColor;

void main() {
    vec3 light_color = vec3(1.0, 1.0, 1.0);

    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light_color;

    vec3 norm = normalize(v_normal);

    vec3 lightDir = normalize(light_pos - v_frag_pos);  
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * light_color;

    vec3 result = (ambient + diffuse) * v_color;
    fragColor = vec4(result, 1.0);

    // fragColor = vec4(originalPosition, 1.0);
    // fragColor = vec4(v_color, 1.0);
}
