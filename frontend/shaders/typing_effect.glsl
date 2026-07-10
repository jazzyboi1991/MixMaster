#ifdef GL_ES
precision mediump float;
#endif

/** @resolution */
uniform vec2 u_resolution;

/** @time */
uniform float u_time;

/**
 * @label Text Color
 * @color
 * @default #ffffff
 */
uniform vec3 u_color;

void main() {
    // 텍스트 바운딩 박스 기준의 정규화 좌표
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    
    // 현대적인 빠른 모션 세팅 (대기 시간 단축, 속도 향상)
    float startDelay = 0.08;
    float speed = 1.35; // 기존보다 2.5배 가량 빨라진 시원하고 쾌적한 속도
    float progress = clamp((u_time - startDelay) * speed, 0.0, 1.0);
    
    // 아날로그 타자기의 뚝뚝 끊기는 느낌(step)을 제거하고,
    // 현대적인 그래디언트 페이드 스위프(Gradient Fade Sweep) 모션 구현
    // 글자가 왼쪽에서 오른쪽으로 슥 훑고 지나가며 부드러운 잔상과 함께 나타납니다.
    float feather = 0.06; // 자연스럽고 부드럽게 번져나오는 구간의 넓이
    float alpha = smoothstep(progress + feather, progress, uv.x);
    
    gl_FragColor = vec4(u_color, alpha);
}
