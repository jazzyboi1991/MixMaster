#ifdef GL_ES
precision mediump float;
#endif

/** @resolution */
uniform vec2 u_resolution;

/** @time */
uniform float u_time;

void main() {
    // 0.0 ~ 1.0 UV 좌표 계산
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    
    // 주기를 정상적인 부드럽고 유기적인 속도(0.35x)로 원복
    float syncTime = u_time * 0.35;
    
    vec2 targetUv = vec2(
        0.5 + 0.28 * cos(syncTime),
        0.5 + 0.18 * sin(syncTime)
    );
    
    // 화면 종횡비를 반영하여 찌그러짐 없는 완벽한 원형 광원 구현
    float aspect = u_resolution.x / u_resolution.y;
    vec2 diff = uv - targetUv;
    diff.x *= aspect;
    float dist = length(diff);
    
    // Elasti 테마 다크 차콜 및 액티브 컬러 정의
    vec3 colorDark = vec3(0.09, 0.07, 0.06);
    vec3 colorActive = vec3(
        0.70 + 0.15 * cos(syncTime),
        0.30 + 0.12 * sin(syncTime),
        0.22 + 0.18 * cos(syncTime)
    );
    
    // 외곽으로 넓고 서서히 흩어지는 감쇠 설정
    float intensity = exp(-dist * 0.70);
    
    // 파동 속도도 자연스럽게 매핑
    float wave = sin(u_time * 0.7) * 0.015;
    float finalGlow = clamp(intensity + wave, 0.0, 1.0);
    
    // 최종 보간 색상 계산
    vec3 finalColor = mix(colorDark, colorActive, finalGlow);
    
    gl_FragColor = vec4(finalColor, 1.0);
}
