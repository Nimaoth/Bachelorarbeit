Shader "Custom/SPolySurface"
{
    Properties
    {
        _StepSize ("Step Size", Range(0.01, 0.5)) = 0.1
        _Color ("Color", Color) = (1, 1, 1)
        _BackfaceColor ("Backface Color", Color) = (0.8, 0.2, 0.2)
        _Bounds ("Bounds", Range(1, 100)) = 50
    }
    SubShader
    {
        Tags {
            "RenderType"="Transparent"
            "Queue"="Transparent+1"
        }

        Pass
        {
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "UnityLightingCommon.cginc"

            #define MAX_DIST 500
            #define MAX_STEPS 500
            #define STEP_SIZE 0.1
            #define SURF_DIST 0.0001

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            struct FragmentOutput {
                fixed4 color : SV_Target;
                half depth : SV_Depth;
            };

            float _StepSize;
            float4 _Color;
            float4 _BackfaceColor;
            float3 cameraPos;
            float3 cameraDir;
            float3 cameraRight;

            float3 cameraPosBL;
            float3 cameraPosBR;
            float3 cameraPosTL;
            float3 cameraPosTR;
            float3 cameraDirBL;
            float3 cameraDirBR;
            float3 cameraDirTL;
            float3 cameraDirTR;

            float cameraFOV;
            float _Bounds;
            float3 center;
            float coefficients[20];

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = o.vertex.xy * 0.5 + 0.5;
                o.uv.y = 1 - o.uv.y;
                return o;
            }

            float sdBox(float3 p, float3 s) {
                p = abs(p)-s;
                return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);
            }

            float sdPoly(float3 p) {
                float x = p.x;
                float y = p.y;
                float z = p.z;
                return
                    coefficients[0] +
                    coefficients[1] * x +
                    coefficients[2] * y +
                    coefficients[3] * z +
                    coefficients[4] * x * x +
                    coefficients[5] * x * y +
                    coefficients[6] * x * z +
                    coefficients[7] * y * y +
                    coefficients[8] * y * z +
                    coefficients[9] * z * z +
                    coefficients[10] * x * x * x +
                    coefficients[11] * x * x * y +
                    coefficients[12] * x * x * z +
                    coefficients[13] * x * y * y +
                    coefficients[14] * x * y * z +
                    coefficients[15] * x * z * z +
                    coefficients[16] * y * y * y +
                    coefficients[17] * y * y * z +
                    coefficients[18] * y * z * z +
                    coefficients[19] * z * z * z;
            }

            float GetDist(float3 p) {
                float box = sdBox(p - center, float3(1, 1, 1) * _Bounds);
                float poly = sdPoly(p - center);
                return max(box, poly);
            }

            float3 GetNormal(float3 pos) {
                // specifically for poly surface
                pos -= center;
                float x = pos.x;
                float y = pos.y;
                float z = pos.z;
                float a = coefficients[0];
                float b = coefficients[1];
                float c = coefficients[2];
                float d = coefficients[3];
                float e = coefficients[4];
                float f = coefficients[5];
                float g = coefficients[6];
                float h = coefficients[7];
                float i = coefficients[8];
                float j = coefficients[9];
                float k = coefficients[10];
                float l = coefficients[11];
                float m = coefficients[12];
                float n = coefficients[13];
                float o = coefficients[14];
                float p = coefficients[15];
                float q = coefficients[16];
                float r = coefficients[17];
                float s = coefficients[18];
                float t = coefficients[19];
                return normalize(float3(
                    2*e*x + 2*l*x*y + 2*m*x*z + 3*k*x*x + b + f*y + g*z + n*y*y + o*y*z + p*z*z,
                    2*h*y + 2*n*x*y + 2*r*y*z + 3*q*y*y + c + f*x + i*z + l*x*x + o*x*z + s*z*z,
                    2*j*z + 2*p*x*z + 2*s*y*z + 3*t*z*z + d + g*x + i*y + m*x*x + o*x*y + r*y*y
                ));
            }

            float BinarySearch(float3 ro, float3 rd, float t_min, float t_max) {
                float d_min = GetDist(ro + rd * t_min);
                float d_max = GetDist(ro + rd * t_max);

                for(int i=0; abs(d_min) >= SURF_DIST && i < MAX_STEPS / 8; i++) {
                    float t = (t_min + t_max) * 0.5;
                    float d = GetDist(ro + rd * t);

                    if (sign(d) == sign(d_min)) {
                        t_min = t;
                    } else {
                        t_max = t;
                    }
                }

                return t_max;
            }

            float2 RayMarch(float3 ro, float3 rd) {
                float stepSize = _StepSize;

                float2 t_prev = float2(0, -1);
                float2 t = float2(stepSize, -1);

                float prev_dist = GetDist(ro);
                float dist;

                for(int i=0; true; i++) {
                    if (i >= MAX_STEPS) {
                        return float2(MAX_DIST + 1.0, t.y);
                    }
                    float3 p = ro + rd * t.x;
                    dist = GetDist(p);

                    if (prev_dist < 0 && dist >= 0) {
                        t_prev.y = t.x - stepSize;
                        t.y = t.x;
                        t.y = BinarySearch(ro, rd, t_prev.y, t.y);
                    }

                    if (prev_dist > 0 && dist <= 0)
                        break;

                    prev_dist = dist;
                    t_prev.x = t.x;
                    t.x += stepSize;
                    stepSize = lerp(_StepSize, _StepSize * 10, t.x / MAX_DIST);
                    // t.x += dist * 0.5;
                    // t.x += clamp(dist, stepSize, stepSize * 1);

                    if (t.x > MAX_DIST)
                        return t;
                }

                t.x = BinarySearch(ro, rd, t_prev.x, t.x);

                return t;
            }

            float3 GetRayDir(float2 uv) {
                float3 dir1 = lerp(cameraDirBL, cameraDirBR, uv.x);
                float3 dir2 = lerp(cameraDirTL, cameraDirTR, uv.x);
                return normalize(lerp(dir1, dir2, uv.y));
            }

            float3 GetRayOrigin(float2 uv) {
                float3 pos1 = lerp(cameraPosBL, cameraPosBR, uv.x);
                float3 pos2 = lerp(cameraPosTL, cameraPosTR, uv.x);
                return lerp(pos1, pos2, uv.y);
            }

            float3 GetColor(float3 pos, float3 color) {
                float3 normal = GetNormal(pos);

                float3 lightDir = _WorldSpaceLightPos0.xyz;
                float3 lightCol = _LightColor0;
                float light = clamp(dot(normal, lightDir), 0.25, 1);
                light = 1;
                // return (normal * 0.5 + 0.5) * color.rgb * light;
                // return normal * color.rgb * light;

                float f = max(0.1, abs(dot(normal, -cameraDir)));
                return color.rgb * f;
            }

            FragmentOutput frag(v2f i)
            {
                // discard;
                FragmentOutput output;
            
                float2 uv = i.vertex.xy / _ScreenParams.xy;
                float3 rayOrigin = GetRayOrigin(uv);
                float3 rayDir = GetRayDir(uv);

                float2 dist = RayMarch(rayOrigin, rayDir);
                if (dist.x >= MAX_DIST && dist.y <= 0) {
                    discard;
                } else if (dist.x < MAX_DIST && dist.y > 0) {
                    float3 col1 = GetColor(cameraPos + rayDir * dist.x, _Color.rgb);
                    float3 col2 = GetColor(cameraPos + rayDir * dist.y, _BackfaceColor.rgb);
                    float3 col = (1 - _BackfaceColor.a) * col1 + _BackfaceColor.a * col2;
                    output.color = float4(col, _Color.a);
                    output.depth = (1.0 / dist.y - _ZBufferParams.w) / _ZBufferParams.z;
                } else if (dist.x < MAX_DIST) {
                    float3 col = GetColor(cameraPos + rayDir * dist.x, _Color.rgb);
                    output.color = float4(col, _Color.a);
                    output.depth = (1.0 / dist.x - _ZBufferParams.w) / _ZBufferParams.z;
                } else if (dist.y > 0) {
                    float3 col = GetColor(cameraPos + rayDir * dist.y, _BackfaceColor.rgb) * _BackfaceColor.a;
                    output.color = float4(col, _Color.a);
                    output.depth = (1.0 / dist.y - _ZBufferParams.w) / _ZBufferParams.z;
                } else {
                    discard;
                }

                return output;
            }
            ENDCG
        }
    }
}
