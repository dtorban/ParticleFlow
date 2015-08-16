#version 330
layout(location = 0) in vec3 vposition;
uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform sampler2D height;

out vec3 txcoord;

void main() {
	vec3 pos = vposition;
	txcoord = pos;
	pos.z = pos.z + texture2D(height, vec2(txcoord.x+0.5, 1.0-txcoord.y+0.5)).x/5.0;
    gl_Position = Projection*View*Model*vec4(pos,1.0);
}