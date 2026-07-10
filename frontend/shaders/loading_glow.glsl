#ifdef GL_ES
precision mediump float;
#endif

/** @resolution */
uniform vec2 u_resolution;

/** @time */
uniform float u_time;

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    
    // 좌우로 흘러가는 광원 진행 위치 연산
    float speed = 2.2;
    float pos = mod(u_time * speed, 2.5) - 0.75;
    
    // 스캔 광선 두께 및 페더(Feather)
    float width = 0.20;
    float scan = smoothstep(pos - width, pos, uv.x) * smoothstep(pos + width, pos, uv.x);
    
    // Elasti 테마 색상 믹싱 (다크 차콜 배경 + 웜 오렌지 광선)
    vec3 baseColor = vec3(0.15, 0.13, 0.12);
    vec3 activeColor = vec3(0.80, 0.408, 0.24);
    
    vec3 color = mix(baseColor, activeColor, scan);
    
    gl_FragColor = vec4(color, 1.0);
}
