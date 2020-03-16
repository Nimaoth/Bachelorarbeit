Shader "Custom/PointCloud"
{
    Properties
    {
        _Color ("Color", Color) = (1, 1, 0, 1)
        _NormalLength ("Normal Length", Range(0.0001, 2)) = 1
    }

    SubShader 
    {
        Tags {
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
        }

        Pass 
        {
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
            };
            
            float4 _Color;
            float _NormalLength;
            StructuredBuffer<Point> pointCloud;

            struct v2g
            {
                float4 vertex : POSITION;
                float4 normal : NORMAL;
            };

            struct g2f
            {
                float4 vertex : SV_POSITION;
                float uv : TEXCOORD0;
            };

            v2g vert(uint instance_id : SV_InstanceID)
            {
                v2g o;

                Point p = pointCloud[instance_id];
                o.vertex = float4(p.pos, 1);
                o.normal = float4(p.normal, 0);

                return o;
            }

            [maxvertexcount(2)]
            void geom(point v2g IN[1], inout LineStream<g2f> stream)
            {
                v2g i = IN[0];

                g2f o;
                o.vertex = UnityWorldToClipPos(i.vertex);
                o.uv = 0;
                stream.Append(o);

                o.vertex = UnityWorldToClipPos(i.vertex + i.normal * _NormalLength * 1);
                o.uv = 1;
                stream.Append(o);
            }

            float4 frag(g2f i) : SV_Target
            {
                return float4(_Color.rgb, 1 - i.uv);
                // return lerp(_Color, 0, i.uv);
                // return lerp(0, _Color, i.uv);
            }
            
            ENDHLSL
        }
    }
}
