

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;


// should have 3 matrices of complex. maybe vectors of length 8


attribute vec3 aVertex;
attribute vec2 aTexCoord;

varying vec2 vTexCoord;

void main(){
   vTexCoord = aTexCoord;
   gl_Position = uPMatrix * uMVMatrix * vec4(aVertex, 1.0);
}
