

uniform sampler2D sTexture;
varying vec2 vTexCoord;

void main(){
   gl_FragColor = texture2D(sTexture, vTexCoord);
}

vec2 complexAdd(vec2 a, vec2 b) {
    return vec2(a.x + b.x, a.y + b.y);
}

vec2 complexMultiply(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}