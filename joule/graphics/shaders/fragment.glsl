#version 330 core

// lighting vector constants
uniform vec3 view_pos;
uniform vec3 light_pos;
uniform vec3 light_color;

// lighting parameter constants
uniform float ambient_strength;
uniform float diffuse_strength;
uniform float diffuse_base;
uniform float specular_strength;
uniform float specular_reflection;

// parameters passed from vertex shader
in vec3 vertex_color;
in vec3 vertex_normal;
in vec3 vertex_frag_pos;

// output pixel color
out vec4 fragColor;

void main() {
    // A HUGE THANK YOU TO: https://learnopengl.com/Lighting/Basic-Lighting
    // THE IDEAS/FORMULAS OF LINES 22 to 36 ARE FROM THAT TUTORIAL
    // quite a bit of modifications were made, though
    // since the current rendering architecture is different

    // ambient lighting
    vec3 ambient = ambient_strength * light_color;

    // diffuse lighting
    vec3 light_dir = normalize(light_pos - vertex_frag_pos);
    float diffuse_component = max(abs(dot(vertex_normal, light_dir)), 0.0);
    vec3 diffuse = sqrt(diffuse_base + diffuse_component * light_color * diffuse_strength);

    // specular lighting
    vec3 view_dir = normalize(view_pos - vertex_frag_pos);
    vec3 reflect_dir = reflect(-light_dir, vertex_normal);
    float ray_align = max(dot(view_dir, reflect_dir), 0.0);
    float specular_component = pow(ray_align, specular_reflection);
    vec3 specular = specular_strength * specular_component * light_color;  

    vec3 combined = (ambient + diffuse + specular) * vertex_color;
    fragColor = vec4(combined, 1.0);
}
