#version 330 core

uniform vec3 light_pos;
uniform vec3 view_pos;

in vec3 v_color;
in vec3 v_normal;
in vec3 v_frag_pos;

out vec4 fragColor;

void main() {
    vec3 light_color = vec3(1.0, 1.0, 1.0);

    vec3 norm = normalize(v_normal);

    float ambient_strength = 0.2;
    vec3 ambient = ambient_strength * (light_color);

    float diffuseStrength = 0.8;
    vec3 lightDir = normalize(light_pos - v_frag_pos);  
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse_ = diff * light_color * diffuseStrength;
    vec3 diffuse = 0.25 + diffuse_ * 0.25;

    float specularStrength = 0.1;
    vec3 viewDir = normalize(view_pos - v_frag_pos);
    vec3 reflectDir = reflect(-lightDir, norm);  

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * light_color;  

    vec3 result = (ambient + sqrt(diffuse) + specular) * v_color;
    fragColor = vec4(result, 1.0);

    // fragColor = vec4(v_normal, 1.0);

    // fragColor = vec4(originalPosition, 1.0);
    // fragColor = vec4(v_color, 1.0);
}
