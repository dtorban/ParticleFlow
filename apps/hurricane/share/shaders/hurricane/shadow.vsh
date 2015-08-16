#version 330
layout(location = 0) in vec3 vposition;
uniform mat4 Shadow;
uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform sampler2D height;
uniform vec3 size;

void main() {
	vec3 pos = vposition;
	pos = (Shadow*vec4(pos,1.0)).xyz;
	vec3 txcoord = pos/size;
	pos.y = pos.y + texture2D(height, txcoord.xz).x/5.0;
    gl_Position = Projection*View*Model*vec4(pos,1.0);
}