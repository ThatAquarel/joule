#version 330 core

uniform vec3 view_pos;
uniform vec3 light_pos;
uniform vec3 light_color;

uniform float ambient_strength;
uniform float diffuse_strength;
uniform float diffuse_base;
uniform float specular_strength;
uniform float specular_reflection;

in vec3 vertex_color;
in vec3 vertex_normal;
in vec3 vertex_frag_pos;

out vec4 fragColor;

void main() {
    vec3 ambient = ambient_strength * light_color;

    vec3 light_dir = normalize(light_pos - vertex_frag_pos);  
    float diffuse_component = max(abs(dot(vertex_normal, light_dir)), 0.0);
    vec3 diffuse = sqrt(diffuse_base + diffuse_component * light_color * diffuse_strength);

    vec3 view_dir = normalize(view_pos - vertex_frag_pos);
    vec3 reflect_dir = reflect(-light_dir, vertex_normal);
    float ray_align = max(dot(view_dir, reflect_dir), 0.0);
    float specular_component = pow(ray_align, specular_reflection);
    vec3 specular = specular_strength * specular_component * light_color;  

    vec3 combined = (ambient + diffuse + specular) * vertex_color;
    fragColor = vec4(combined, 1.0);
}
