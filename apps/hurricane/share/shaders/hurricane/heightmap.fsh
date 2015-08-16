#version 330
layout(location = 0) out vec4 FragColor;
uniform sampler2D height;

in vec3 txcoord;

void main() {
	vec4 val = texture2D(height, txcoord.yx+0.5);
	if (val.x > 0.0)
	{
		FragColor = vec4(val.x,0.0,0.0,1.0);
	}
	else
	{
		FragColor = vec4(0.0,0.0,0.4,1.0);
	}
}