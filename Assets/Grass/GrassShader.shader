Shader "Unlit/GrassShader"
{
    Properties
    {
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull off

        Pass
        {
            CGPROGRAM
            #pragma target 5.0
            #pragma require tessellation tessHW
            #pragma require geometry
            #pragma vertex vertexShader
            #pragma fragment fragmentShader
            #pragma geometry geometryShader
            #pragma hull hullShader
            #pragma domain domainShader
            #define PI 3.141592653589793


            #include "UnityCG.cginc"

            struct VertexIn
            {
                float4 position : POSITION;
            };

            struct VertexOut
            {
                float4 position : SV_POSITION;
            };

            struct GeometryOut
            {
                float4 position : SV_POSITION;
                float4 color : float4;
            };

            struct TessellationFactors {
                float edge[3]: SV_TessFactor;
                float inside: SV_InsideTessFactor;
            };

            struct DomainOut {
                float4 position: SV_Position;
                float3 normal: float3;
            };


            uniform float3 _PlayerPosition;

            float3 normalFrom3Points(float3 point1, float3 point2, float3 point3) {
                float3 pq = point2 - point1;
                float3 pr = point3 - point1;
                return normalize(cross(pq, pr));
            }


            VertexOut vertexShader (VertexIn vertIn)
            {
                VertexOut vertOut;
                vertOut.position = vertIn.position;
                return vertOut;
            }
            
            TessellationFactors patchConstantFunction(
                InputPatch<VertexOut, 3> patch
            ) {
                //UNITY_SETUP_INSTANCE_ID(patch[0]);
                TessellationFactors f;
                f.edge[0] = 1;
                f.edge[1] = 1;
                f.edge[2] = 1;

                if (distance(_PlayerPosition, patch[0].position) > 70 &&
                    distance(_PlayerPosition, patch[1].position) > 70 &&
                    distance(_PlayerPosition, patch[2].position) > 70
                ) {
                    f.edge[0] = 0;
                }

                f.inside = 20;
                return f;
            }

            [domain("tri")]
            [outputcontrolpoints(3)]
            [outputtopology("triangle_ccw")]
            [patchconstantfunc("patchConstantFunction")]
            [partitioning("integer")]
            VertexOut hullShader(
                InputPatch<VertexOut, 3> patch, // In triangle,
                uint id: SV_OutputControlPointID)
            {
                return patch[id];
            }

            #define BARYCENTRIC_INTERPOLATE(fieldName) \
                patch[0].fieldName * barycentricCoords.x + \
                patch[1].fieldName * barycentricCoords.y + \
                patch[2].fieldName * barycentricCoords.z

            [domain("tri")]
            DomainOut domainShader(
                TessellationFactors factors,
                OutputPatch<VertexOut, 3> patch,
                float3 barycentricCoords: SV_DomainLocation
            ) {
                DomainOut output;

                output.position = BARYCENTRIC_INTERPOLATE(position);
                output.normal = normalFrom3Points(patch[0].position.xyz, patch[1].position.xyz, patch[2].position.xyz);

                return output;
            }

            GeometryOut setupVertex(float3 position, float4 color) {
                GeometryOut o;
                o.position = UnityObjectToClipPos(float4(position, 1));
                o.color = color;
                return o;
            }

            // Generates a random number
            float rand(float2 uv)
            {
                return frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
            }

            float hash(float n)
            {
                return frac(sin(n)*43758.5453);
            }

            float noise( float3 x )
            {
                // The noise function returns a value in the range -1.0f -> 1.0f
                x += _Time;
                float3 p = floor(x);
                float3 f = frac(x);

                f = f*f*(3.0-2.0*f);
                float n = p.x + p.y*57.0 + 113.0*p.z;

                return lerp(lerp(lerp( hash(n+0.0), hash(n+1.0),f.x),
                            lerp( hash(n+57.0), hash(n+58.0),f.x),f.y),
                        lerp(lerp( hash(n+113.0), hash(n+114.0),f.x),
                            lerp( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
            }


            float3x3 getRotationMatrix(float degrees) {
                degrees = degrees * PI / 180;       
                return float3x3(cos(degrees), 0 , -sin(degrees), 0, 1, 0, sin(degrees), 0, cos(degrees));
            }
            //https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724
            float3x3 rotateAlign( float3 v1, float3 v2)
            {
                float3 axis = cross( v1, v2 );

                const float cosA = dot( v1, v2 );
                const float k = 1.0f / (1.0f + cosA);

                float3x3 result = float3x3(
                    (axis.x * axis.x * k) + cosA,
                    (axis.x * axis.y * k) + axis.z,  
                    (axis.x * axis.z * k) - axis.y,  
                    
                    (axis.y * axis.x * k) - axis.z, 
                    (axis.y * axis.y * k) + cosA,      
                    (axis.y * axis.z * k) + axis.x,  
                    
                    (axis.z * axis.x * k) + axis.y,
                    (axis.z * axis.y * k) - axis.x,
                    (axis.z * axis.z * k) + cosA);
                return result;
            }

            float3x3 calcBladeHalfWidth(float y) {
                float a = 0.2;
                float b = 2.5;
                float t = -0.13;
                return a*(y-t) * exp(-b*(y-t));
            }

            float3x3 calcNaturalCurvature(float y) {
                float a = 0.15;
                float b = 0.5;
                float t = -0.13;
                return a*y*exp((y-t));
            }
            
            float3 calcSegmentVector(
                float3 position,
                float3 normal,
                float3x3 rotationMatrix,
                float segmentLength,
                float fullLengthPercent
            ) {
                // Do not do further calculations if it is the root this will need to change to account for normals
                if (fullLengthPercent == 0.0) {
                    return float3(0, 0, 0);
                }
                float rand1 = rand(position.xz);
                float rand2 = rand(float2(rand1, position.x));

                float naturalCurvature = calcNaturalCurvature(fullLengthPercent * rand2);
                float3 bladeSegment = float3(0, segmentLength, -naturalCurvature);
                bladeSegment = mul(rotationMatrix, bladeSegment);

                float3x3 modelRotMatrix = rotateAlign(normal, float3(0,1,0));
                bladeSegment = segmentLength * normalize(mul(modelRotMatrix,bladeSegment));
                

                float3 bladePosition = bladeSegment + position;
                float3 bladeToPlayer = _PlayerPosition - bladePosition;
                float distToPlayer = length(bladeToPlayer);
                float3 playerPush = -normalize(bladeToPlayer) * 1/distToPlayer * fullLengthPercent; 

                float windXStrength = (noise(position.xyz + _Time.y * 0.3) * 2 - 1) * 0.3;
                float windYStrength = (noise(position.yzx + _Time.y * 0.3) * 2 - 1) * 0.3;
                float windZStrength = (noise(position.zxy + _Time.y * 0.3) * 2 - 1) * 0.3;
                float3 wind = float3(windXStrength, windYStrength, windZStrength) * fullLengthPercent;
                bladeSegment += wind;
                //bladeSegment += playerPush;
                return segmentLength * normalize(bladeSegment);
            }


            [maxvertexcount(11)]
            void geometryShader (triangle DomainOut inputs[3], inout TriangleStream<GeometryOut> outputStream)
            {
                float _NUM_SEGMENTS = 4.0;
                float _MAX_BLADE_LENGTH = 1.0;
                float _MIN_BLADE_LENGTH = 0.4;


                for (int i = 0; i < 3; i++) {
                    float3 position = inputs[i].position.xyz;
                    float3 normal = inputs[i].normal;
                    float rand1 = rand(position.xz);
                    float rand2 = rand(float2(rand1, position.x));
                    
                    float3x3 rotationMatrix = getRotationMatrix(rand1 * 360);
                    float3 grassFwdDir = mul(rotationMatrix, float3(0, 0, 1));

                    float fullBladeLength = lerp(_MIN_BLADE_LENGTH, _MAX_BLADE_LENGTH, rand1);
                    // Calculate blade length;
                    float3 root = position;
                    for (float i = 0.0; i < _NUM_SEGMENTS; i+=1.0) {
                        float fullLengthPercent = i / _NUM_SEGMENTS;
                        float segmentLength = fullLengthPercent * fullBladeLength;
                        float halfWidth = calcBladeHalfWidth(fullLengthPercent);
                        // Grass Generation
                        float3 bladeSegment = calcSegmentVector(position, normal, rotationMatrix, segmentLength, fullLengthPercent);
                        // Calculate the segment normalised left and right vectors
                        // The grass should face the grassFwdDir
                        float3 widthDir;
                        if (i == 0.0) {
                            widthDir = normalize(cross(float3(0,1,0), grassFwdDir));
                        }
                        else {
                            widthDir = normalize(cross(bladeSegment, grassFwdDir));
                        }
                        // Calculate blade left and right in local spaces
                        float3 bladeLeft = bladeSegment + widthDir * halfWidth;
                        float3 bladeRight = bladeSegment - widthDir * halfWidth;
                        // Calculate Normal
                        //float3 normal;
                        // Sets the roots for the next segmentLength
                        //root;
                        // Emits the vertex
                        float4 color = lerp(float4(130,190,64,255) / 255, float4(181,233,97,255) / 255, fullLengthPercent);
                        outputStream.Append(setupVertex(position + bladeLeft , color));
                        outputStream.Append(setupVertex(position + bladeRight, color));
                    }
                    // Calculate tip position
                    outputStream.Append(setupVertex(position + calcSegmentVector(position, normal, rotationMatrix, fullBladeLength, 1), float4(198,242,104,255) / 255));
                    outputStream.RestartStrip();
                }
            }

            fixed4 fragmentShader (GeometryOut fragIn) : SV_Target
            {
                float4 fragOut = fragIn.color;
                return fragOut;
            }
            ENDCG
        }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = float4(182,215,126,0) / 255;
                return col;
            }
            ENDCG
        }
    }
}
