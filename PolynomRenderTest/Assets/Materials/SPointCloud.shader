Shader "Custom/PointCloud"
{
    Properties
    {
        _Color ("Color", Color) = (1, 1, 0, 1)
        _Weight0 ("Weight 0", Color) = (0.1, 0.1, 0.9, 1)
        _Weight1 ("Weight 1", Color) = (0.9, 0.1, 0.1, 1)
        _NormalLength ("Normal Length", Range(0.0001, 5)) = 1
    }

    SubShader 
    {
        Tags {
            "RenderType" = "Transparent"
            "Queue" = "Transparent+2"
        }

        Pass 
        {
            ZTest Always
            // Blend One One
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            
            #include "UnityCG.cginc"
            
            //The same particle data we're using in the compute shader
            struct Point
            {
                float3 pos;
                float3 normal;
                float weight;
            };
            
            float3 center;
            float4 _Color;
            float _WeightFactor;
            float _WeightScale;
            float4 _Weight0;
            float4 _Weight1;
            float _NormalLength;
            StructuredBuffer<Point> pointCloud;

            struct v2g
            {
                float4 vertex : POSITION;
                float4 normal : NORMAL;
                float weight  : TEXCOORD0;
            };

            struct g2f
            {
                float4 vertex   : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float uv        : TEXCOORD1;
                float weight    : TEXCOORD2;
            };

            v2g vert(uint instance_id : SV_InstanceID)
            {
                v2g o;

                Point p = pointCloud[instance_id];
                o.vertex = float4(p.pos, 1);
                o.normal = float4(p.normal, 0);
                o.weight = p.weight;

                return o;
            }

            [maxvertexcount(2)]
            void geom(point v2g IN[1], inout LineStream<g2f> stream)
            {
                v2g i = IN[0];

                g2f o;
                o.weight = i.weight;

                o.vertex = UnityWorldToClipPos(i.vertex);
                o.worldPos = i.vertex.xyz;
                o.uv = 0;
                stream.Append(o);

                o.vertex = UnityWorldToClipPos(i.vertex + i.normal * _NormalLength);
                o.worldPos = i.vertex.xyz + i.normal * _NormalLength;
                o.uv = 1;
                stream.Append(o);
            }

            float GetWeight(float r) {
                return lerp(1, exp(-0.5 * r * _WeightScale), _WeightFactor);
            }

            float4 frag(g2f i) : SV_Target
            {
                float4 color = _Color;

                float dist = distance(i.worldPos, center);
                if (dist <= -2 * (log(0.1)) / _WeightScale) {
                    float weight = GetWeight(dist) * i.weight;
                    color = lerp(_Weight0, _Weight1, weight);
                }
                color.a = 1 - i.uv;
                return color;
            }
            
            ENDHLSL
        }
    }
}
