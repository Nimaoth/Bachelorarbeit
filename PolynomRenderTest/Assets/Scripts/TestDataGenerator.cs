using System;
using System.Collections;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Single;
using Newtonsoft.Json.Converters;
using System.IO;
using Newtonsoft.Json;


[Serializable]
public struct vec3
{
    public float x, y, z;

    public vec3(float x, float y, float z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}

[Serializable]
public struct Sample {
    public vec3 entryPoint;
    public vec3 exitPoint;
    public float[] coefficients;
    public float sigmaS;
    public float sigmaA;
    public float sigmaT;
    public float g;
    public float ior;
}

public class TestDataGenerator : MonoBehaviour
{
    private static MeshCollider meshCollider;

    public static void GenerateTestData(
            Mesh mesh,
            int minimumPointCount,
            float pointsPerAreaMultiplier,
            float weightFactor,
            float weightScale,
            int maxVertexCount,
            int randomVertexCount,
            float randomVertexWeight,
            float largeCoefficientPenalty,
            int randomVertexSeed,
            int sampleCount,
            string outputPath) {

        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Restart();
        
        // generate point cloud
        var pointCloud = ComputePointCloud(mesh, minimumPointCount, pointsPerAreaMultiplier, 0.1f);

        // get random point on surface + normal
        Vector3[] meshVertices = mesh.vertices;
        Vector3[] meshNormals = mesh.normals;
        int[] meshTriangleIndices = mesh.triangles;

        if (meshCollider == null) {
            var tempObject = new GameObject();
            meshCollider = tempObject.AddComponent<MeshCollider>();
        }
        meshCollider.sharedMesh = mesh;

        var samples = new List<Sample>(sampleCount);

        for (int i = 0; i < sampleCount; i++) {

            // generate random point and normal on surface
            int randomTriangleIndex = UnityEngine.Random.Range(0, meshTriangleIndices.Length / 3) * 3;
            var (b0, b1, b2) = TestDataGenerator.GetRandomBarycentric();
            Vector3 randomSurfacePoint =
                b0 * meshVertices[meshTriangleIndices[randomTriangleIndex + 0]] +
                b1 * meshVertices[meshTriangleIndices[randomTriangleIndex + 1]] +
                b2 * meshVertices[meshTriangleIndices[randomTriangleIndex + 2]];
            Vector3 randomSurfacePointNormal =
                b0 * meshNormals[meshTriangleIndices[randomTriangleIndex + 0]] +
                b1 * meshNormals[meshTriangleIndices[randomTriangleIndex + 1]] +
                b2 * meshNormals[meshTriangleIndices[randomTriangleIndex + 2]];
            randomSurfacePointNormal = randomSurfacePointNormal.normalized;

            Vector3 normalCSRight = GetPerpendicular(randomSurfacePointNormal);
            Vector3 normalCSUp = Vector3.Cross(randomSurfacePointNormal, normalCSRight);
            Vector3 randomNormalTemp = UnityEngine.Random.onUnitSphere;
            randomNormalTemp.z = Mathf.Abs(randomNormalTemp.z);
            randomNormalTemp = randomNormalTemp.x * normalCSRight + randomNormalTemp.y * normalCSUp + randomNormalTemp.z * randomSurfacePointNormal;

            Vector3 direction = -randomNormalTemp.normalized;
            Vector3 origin = randomSurfacePoint - direction * 0.1f;

            // generate random surface properties
            float g = UnityEngine.Random.Range(0.1f, 0.95f);
            float ior = UnityEngine.Random.Range(1.0f, 1.5f);
            float sigmaS = UnityEngine.Random.Range(0.5f, 50.0f);
            float sigmaT = UnityEngine.Random.Range(sigmaS + 0.1f, 51.0f);
            float sigmaA = sigmaT - sigmaS;
            float sigmaSRed = (1 - g) * sigmaS;

            float standardDeviation = 0.0f;
            {
                float sigmaTred = sigmaA + sigmaSRed;
                float alphaRed = sigmaSRed / sigmaTred;

                float e8 = Mathf.Pow((float)Math.E, 8);
                float alphaEff = 1 - (0.125f * Mathf.Log(e8 + alphaRed * (1 - e8)));

                float mad = MAD(g, alphaRed);

                standardDeviation = 2 * mad / sigmaTred;
            }

            // calculate coefficients for random point on surfice
            float[] coefficients = TestDataGenerator.CalculateCoefficients(
                pointCloud.vertices,
                randomSurfacePoint,
                standardDeviation,
                weightFactor,
                weightScale,
                maxVertexCount,
                randomVertexCount,
                randomVertexWeight,
                largeCoefficientPenalty,
                randomVertexSeed);

            SimulatePath(coefficients, origin, direction, randomSurfacePoint, g, sigmaSRed);

            samples.Add(new Sample{
                entryPoint      = new vec3(randomSurfacePoint.x, randomSurfacePoint.y, randomSurfacePoint.z),
                exitPoint       = new vec3(0, 0, 0), // @todo
                coefficients    = coefficients,
                sigmaS          = sigmaS,
                sigmaA          = sigmaA,
                sigmaT          = sigmaT,
                g               = g,
                ior             = ior
            });
        }

        var jsonString = JsonConvert.SerializeObject(samples.ToArray(), Formatting.Indented);
        Debug.Log(jsonString);
        File.WriteAllText("samples.json", jsonString);

        var time = stopwatch.Elapsed;

        Debug.Log($"Generating {sampleCount} samples took {time}");
    }

    private static float MAD(float g, float a_red) {
        float e8 = Mathf.Pow((float)Math.E, 8);
        float a_eff = 1 - (0.125f * Mathf.Log(e8 + a_red * (1 - e8)));
        return 0.25f * g + 0.25f * a_red + a_eff;
    }

    private static void SimulatePath(float[] coefficients, Vector3 origin, Vector3 direction, Vector3 center, float g, float sigmaSRed) {
        direction = direction.normalized;

        float maxDist = float.MaxValue;

        for (int i = 0; i < 500; i++) {
            var (dist, inside) = PolySurface.RayMarch(coefficients, origin, direction, maxDist, center);
            var newPoint = origin + direction * dist;
            var point = new PathPoint{
                position = newPoint,
                direction = direction
            };

            var normal = PolySurface.GetNormalAt(coefficients, newPoint, center);
            origin = newPoint - normal * PolySurface.SURF_DIST * 2;

            if (!inside) {
                // @todo: refract
                var (_dist, _) = PolySurface.RayMarch(coefficients, origin, direction, 10.0f, center);
                // currentPath.Add(point);
                // currentPath.Add(new PathPoint{
                //     position = origin + direction * 10,
                // });
                break;
            }

            // still inside, set random direction and maxDist
            if (g == 0) {
                direction = UnityEngine.Random.onUnitSphere;
            } else {
                float rand = UnityEngine.Random.Range(0.0f, 1.0f);
                float cosAngle = InvIntPhaseFunction(rand, g);
                float theta = Mathf.Acos(cosAngle);
                float d = 1 / Mathf.Tan(theta);

                // Debug.Log($"rand: {rand}, cosAngle: {cosAngle}, theta: {theta}, d: {d}");

                if (d == float.PositiveInfinity) {

                } else if (d == float.NegativeInfinity) {
                    direction = -direction;
                } else {
                    var right = GetPerpendicular(direction).normalized;
                    var up = Vector3.Cross(right, direction).normalized;
                    var uv = UnityEngine.Random.insideUnitCircle;

                    point.uv = uv;
                    point.right = right;
                    point.up = up;

                    // Debug.Log($"right: {right}, up: {up}, uv: {uv}");

                    direction = uv.x * right + uv.y * up + d * direction;
                    direction = direction.normalized;
                }
            }

            // Debug.Log($"direction: {direction}");

            // direction = UnityEngine.Random.onUnitSphere;
            maxDist = GetScatterDistance(sigmaSRed);
            point.direction = point.direction * Vector3.Dot(point.direction, direction * maxDist);
            // currentPath.Add(point);
        }
    }

    private static float GetScatterDistance(float sigmaSRed) {
        float dist = -Mathf.Log(UnityEngine.Random.Range(0, 1)) / sigmaSRed;
        return dist;
    }

    private static float InvIntPhaseFunction(float x, float g) {
        float g2 = g * g;

        float tmp = (1 - g2) / (1 + g * (2 * x - 1));
        return (1 / (2 * g)) * (1 + g2 - tmp * tmp);
    }

    public static PointCloud ComputePointCloud(Mesh mesh, int minimumPointCount, float pointsPerAreaMultiplier, float standardDeviation) {
        Vector3[] positions = mesh.vertices;
        Vector3[] normals = mesh.normals;
        int[] triangles = mesh.GetTriangles(0);
        // Debug.Log($"Mesh {mesh.name}: {positions.Length} verts, {normals.Length} normals, {triangles.Length} triangles");

        List<Vertex> vertices = new List<Vertex>();

        // compute total surface area
        float totalSurfaceArea = 0.0f;
        for (int i = 0; i < triangles.Length; i += 3) {
            int i1 = triangles[i];
            int i2 = triangles[i + 1];
            int i3 = triangles[i + 2];
            Vector3 v1 = positions[i1];
            Vector3 v2 = positions[i2];
            Vector3 v3 = positions[i3];

            totalSurfaceArea += AreaOfTriangle(v1, v2, v3);
        }

        var pointsPerArea = (int)(2 * Mathf.Pow(standardDeviation, -2) * pointsPerAreaMultiplier);
        int pointCount = Mathf.Max(minimumPointCount, (int)totalSurfaceArea * pointsPerArea);
        pointCount = Mathf.Min(pointCount, 100_000);

        for (int i = 0; i < triangles.Length; i += 3) {
            int i1 = triangles[i];
            int i2 = triangles[i + 1];
            int i3 = triangles[i + 2];
            Vector3 v1 = positions[i1];
            Vector3 v2 = positions[i2];
            Vector3 v3 = positions[i3];
            Vector3 n1 = normals[i1];
            Vector3 n2 = normals[i2];
            Vector3 n3 = normals[i3];

            float area = AreaOfTriangle(v1, v2, v3);
            double pointsInTriangle = (double)area / (double)totalSurfaceArea * pointCount;

            if (UnityEngine.Random.value < (pointsInTriangle - (int)pointsInTriangle)) {
                pointsInTriangle += 1;
            }
            // Debug.Log($"Tri {i / 3}: area = {area} ({area/totalSurfaceArea*100}% of total), generating {(int)pointsInTriangle} points");

            for (int k = 0; k < (int)pointsInTriangle; k++) {
                var (a, b, c) = GetRandomBarycentric();

                Vector3 pos = (a * v1 + b * v2 + c * v3);
                Vector3 normal = (a * n1 + b * n2 + c * n3).normalized;
                vertices.Add(new Vertex(pos, normal));
            }
        }

        // Debug.Log($"Generated {vertices.Count} points");
        return new PointCloud(mesh) {
            vertices = vertices.ToArray()
        };
    }

    public static (float, float, float) GetRandomBarycentric() {
        float r1 = Mathf.Sqrt(UnityEngine.Random.Range(0.0f, 1.0f));
        float r2 = UnityEngine.Random.Range(0.0f, 1.0f);
        float a = 1 - r1;
        float b = r1 * (1 - r2);
        float c = r1 * r2;
        return (a, b, c);
    }

    public static float[] CalculateCoefficients(Vertex[] vertices, Vector3 center, float standardDeviation, float weightFactor, float weightScale, int maxVertexCount, int randomVertexCount, float randomVertexWeight, float largeCoefficientPenalty, int randomVertexSeed) {
        var random = new System.Random(randomVertexSeed);

        float sigmaNInv = 1.0f / standardDeviation;

        float maxDist = Mathf.Pow(-2 * Mathf.Log(0.01f) / weightScale, 2);
        vertices = vertices
            .Where(v => (v.position - center).sqrMagnitude <= maxDist)
            .OrderBy(v => (v.position - center).sqrMagnitude)
            .Take(Mathf.Min(vertices.Length, maxVertexCount))
            .Concat(Enumerable.Repeat(0, randomVertexCount).Select(_ => vertices[random.Next() % vertices.Length].withWeight(randomVertexWeight)))
            .Select(v => new Vertex((v.position - center) * sigmaNInv, v.normal, v.weight))
            .ToArray();

        var X = DenseMatrix.OfRows(vertices.Length, 20, vertices.Select(v => Pc(v, weightFactor, weightScale)));
        var tmp = 2 * X.Transpose() * X + Sum(20, 20, vertices, v => weight(v, weightFactor, weightScale) * Y(v.position)) + 2 * largeCoefficientPenalty;
        var c = tmp.Inverse() * Sum(20, 1, vertices, v => weight(v, weightFactor, weightScale) * Z(v.position, v.normal));

        var coefficients = c.Column(0).ToArray();
        coefficients[0] = 0;
        return coefficients;
    }

    private static Vector3 GetPerpendicular(Vector3 p) {
        var t = new Vector3(p.z, p.z, -p.x - p.y);
        if (t.sqrMagnitude < 0.01f) {
            return new Vector3(-p.y - p.z, p.x, p.x);
        } else {
            return t;
        }
    }

    private static float AreaOfTriangle(Vector3 v1, Vector3 v2, Vector3 v3) {
        var a = v2 - v1;
        var b = v3 - v1;
        return Vector3.Cross(a, b).magnitude * 0.5f;
    }

    private static DenseMatrix Sum(int r, int c, Vertex[] vertices, Func<Vertex, DenseMatrix> func) {
        DenseMatrix result = new DenseMatrix(r, c);
        for (int i = 0; i < vertices.Length; i++) {
            result += func(vertices[i]);
        }
        return result;
    }

    private static float weight(Vertex v, float weightFactor, float weightScale) {
        float weight = v.weight;
        if (weight < 0)
            weight = Mathf.Exp(-0.5f * v.position.sqrMagnitude * weightScale);
        return Mathf.Lerp(1, weight, weightFactor);
    }

    private static float[] Pc(Vertex v, float weightFactor, float weightScale) {
        float w = Mathf.Sqrt(weight(v, weightFactor, weightScale));
        Vector3 p = v.position;
        return new float[] {
            w * 1,
            w * p.x,
            w * p.y,
            w * p.z,
            w * p.x * p.x,
            w * p.x * p.y,
            w * p.x * p.z,
            w * p.y * p.y,
            w * p.y * p.z,
            w * p.z * p.z,
            w * p.x * p.x * p.x,
            w * p.x * p.x * p.y,
            w * p.x * p.x * p.z,
            w * p.x * p.y * p.y,
            w * p.x * p.y * p.z,
            w * p.x * p.z * p.z,
            w * p.y * p.y * p.y,
            w * p.y * p.y * p.z,
            w * p.y * p.z * p.z,
            w * p.z * p.z * p.z,
        };
    }

    private static DenseMatrix Z(Vector3 p, Vector3 n) {
        float x = p.x;
        float y = p.y;
        float z = p.z;
        float u = n.x;
        float v = n.y;
        float w = n.z;
        float ux = u*x;
        float uy = u*y;
        float uz = u*z;
        float vx = v*x;
        float vy = v*y;
        float vz = v*z;
        float wx = w*x;
        float wy = w*y;
        float wz = w*z;
        float uxx = u*x*x;
        float uxy = u*x*y;
        float uxz = u*x*z;
        float uyy = u*y*y;
        float uyz = u*y*z;
        float uzz = u*z*z;
        float vxx = v*x*x;
        float vxy = v*x*y;
        float vxz = v*x*z;
        float vyy = v*y*y;
        float vyz = v*y*z;
        float vzz = v*z*z;
        float wxx = w*x*x;
        float wxy = w*x*y;
        float wxz = w*x*z;
        float wyy = w*y*y;
        float wyz = w*y*z;
        float wzz = w*z*z;
        return DenseMatrix.OfArray(new float[,] {
            { 0 },
            { 2 * u },
            { 2 * v },
            { 2 * w },
            { 4 * ux },
            { 2 * uy + 2 * vx },
            { 2 * uz + 2 * wx },
            { 4 * vy },
            { 2 * vz + 2 * wy },
            { 4 * wz },
            { 6 * uxx },
            { 2 * vxx + 4 * uxy },
            { 2 * wxx + 4 * uxz },
            { 2 * uyy + 4 * vxy },
            { 2 * uyz + 2 * vxz + 2 * wxy },
            { 2 * uzz + 4 * wxz },
            { 6 * vyy },
            { 2 * wyy + 4 * vyz },
            { 2 * vzz + 4 * wyz },
            { 6 * wzz },
        });
    }

    private static DenseMatrix Y(Vector3 p) {
        float x = p.x;
        float y = p.y;
        float z = p.z;
        float xx = x * x;
        float xy = x * y;
        float xz = x * z;
        float yy = y * y;
        float yz = y * z;
        float zz = z * z;
        float xxx = x * x * x;
        float xxy = x * x * y;
        float xxz = x * x * z;
        float xyy = x * y * y;
        float xyz = x * y * z;
        float xzz = x * z * z;
        float yyy = y * y * y;
        float yyz = y * y * z;
        float yzz = y * z * z;
        float zzz = z * z * z;
        float xxxx = x * x * x * x;
        float xxxy = x * x * x * y;
        float xxxz = x * x * x * z;
        float xxyy = x * x * y * y;
        float xxyz = x * x * y * z;
        float xxzz = x * x * z * z;
        float xyyy = x * y * y * y;
        float xyyz = x * y * y * z;
        float xyzz = x * y * z * z;
        float xzzz = x * z * z * z;
        float yyyy = y * y * y * y;
        float yyyz = y * y * y * z;
        float yyzz = y * y * z * z;
        float yzzz = y * z * z * z;
        float zzzz = z * z * z * z;
        return DenseMatrix.OfArray(new float[,]{
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 
            { 0, 2, 0, 0, 4 * x, 2 * y, 2 * z, 0, 0, 0, 6 * xx, 4 * xy, 4 * xz, 2 * yy, 2 * yz, 2 * zz, 0, 0, 0, 0 }, 
            { 0, 0, 2, 0, 0, 2 * x, 0, 4 * y, 2 * z, 0, 0, 2 * xx, 0, 4 * xy, 2 * xz, 0, 6 * yy, 4 * yz, 2 * zz, 0 }, 
            { 0, 0, 0, 2, 0, 0, 2 * x, 0, 2 * y, 4 * z, 0, 0, 2 * xx, 0, 2 * xy, 4 * xz, 0, 2 * yy, 4 * yz, 6 * zz }, 
            { 0, 4 * x, 0, 0, 8 * xx, 4 * xy, 4 * xz, 0, 0, 0, 12 * xxx, 8 * xxy, 8 * xxz, 4 * xyy, 4 * xyz, 4 * xzz, 0, 0, 0, 0 }, 
            { 0, 2 * y, 2 * x, 0, 4 * xy, 2 * yy + 2 * xx, 2 * yz, 4 * xy, 2 * xz, 0, 6 * xxy, 4 * xyy + 2 * xxx, 4 * xyz, 2 * yyy + 4 * xxy, 2 * yyz + 2 * xxz, 2 * yzz, 6 * xyy, 4 * xyz, 2 * xzz, 0 }, 
            { 0, 2 * z, 0, 2 * x, 4 * xz, 2 * yz, 2 * zz + 2 * xx, 0, 2 * xy, 4 * xz, 6 * xxz, 4 * xyz, 4 * xzz + 2 * xxx, 2 * yyz, 2 * yzz + 2 * xxy, 2 * zzz + 4 * xxz, 0, 2 * xyy, 4 * xyz, 6 * xzz }, 
            { 0, 0, 4 * y, 0, 0, 4 * xy, 0, 8 * yy, 4 * yz, 0, 0, 4 * xxy, 0, 8 * xyy, 4 * xyz, 0, 12 * yyy, 8 * yyz, 4 * yzz, 0 }, 
            { 0, 0, 2 * z, 2 * y, 0, 2 * xz, 2 * xy, 4 * yz, 2 * zz + 2 * yy, 4 * yz, 0, 2 * xxz, 2 * xxy, 4 * xyz, 2 * xzz + 2 * xyy, 4 * xyz, 6 * yyz, 4 * yzz + 2 * yyy, 2 * zzz + 4 * yyz, 6 * yzz }, 
            { 0, 0, 0, 4 * z, 0, 0, 4 * xz, 0, 4 * yz, 8 * zz, 0, 0, 4 * xxz, 0, 4 * xyz, 8 * xzz, 0, 4 * yyz, 8 * yzz, 12 * zzz }, 
            { 0, 6 * xx, 0, 0, 12 * xxx, 6 * xxy, 6 * xxz, 0, 0, 0, 18 * xxxx, 12 * xxxy, 12 * xxxz, 6 * xxyy, 6 * xxyz, 6 * xxzz, 0, 0, 0, 0 }, 
            { 0, 4 * xy, 2 * xx, 0, 8 * xxy, 4 * xyy + 2 * xxx, 4 * xyz, 4 * xxy, 2 * xxz, 0, 12 * xxxy, 8 * xxyy + 2 * xxxx, 8 * xxyz, 4 * xyyy + 4 * xxxy, 4 * xyyz + 2 * xxxz, 4 * xyzz, 6 * xxyy, 4 * xxyz, 2 * xxzz, 0 }, 
            { 0, 4 * xz, 0, 2 * xx, 8 * xxz, 4 * xyz, 4 * xzz + 2 * xxx, 0, 2 * xxy, 4 * xxz, 12 * xxxz, 8 * xxyz, 8 * xxzz + 2 * xxxx, 4 * xyyz, 4 * xyzz + 2 * xxxy, 4 * xzzz + 4 * xxxz, 0, 2 * xxyy, 4 * xxyz, 6 * xxzz }, 
            { 0, 2 * yy, 4 * xy, 0, 4 * xyy, 2 * yyy + 4 * xxy, 2 * yyz, 8 * xyy, 4 * xyz, 0, 6 * xxyy, 4 * xyyy + 4 * xxxy, 4 * xyyz, 2 * yyyy + 8 * xxyy, 2 * yyyz + 4 * xxyz, 2 * yyzz, 12 * xyyy, 8 * xyyz, 4 * xyzz, 0 }, 
            { 0, 2 * yz, 2 * xz, 2 * xy, 4 * xyz, 2 * yyz + 2 * xxz, 2 * yzz + 2 * xxy, 4 * xyz, 2 * xzz + 2 * xyy, 4 * xyz, 6 * xxyz, 4 * xyyz + 2 * xxxz, 4 * xyzz + 2 * xxxy, 2 * yyyz + 4 * xxyz, 2 * yyzz + 2 * xxzz + 2 * xxyy, 2 * yzzz + 4 * xxyz, 6 * xyyz, 4 * xyzz + 2 * xyyy, 2 * xzzz + 4 * xyyz, 6 * xyzz }, 
            { 0, 2 * zz, 0, 4 * xz, 4 * xzz, 2 * yzz, 2 * zzz + 4 * xxz, 0, 4 * xyz, 8 * xzz, 6 * xxzz, 4 * xyzz, 4 * xzzz + 4 * xxxz, 2 * yyzz, 2 * yzzz + 4 * xxyz, 2 * zzzz + 8 * xxzz, 0, 4 * xyyz, 8 * xyzz, 12 * xzzz }, 
            { 0, 0, 6 * yy, 0, 0, 6 * xyy, 0, 12 * yyy, 6 * yyz, 0, 0, 6 * xxyy, 0, 12 * xyyy, 6 * xyyz, 0, 18 * yyyy, 12 * yyyz, 6 * yyzz, 0 }, 
            { 0, 0, 4 * yz, 2 * yy, 0, 4 * xyz, 2 * xyy, 8 * yyz, 4 * yzz + 2 * yyy, 4 * yyz, 0, 4 * xxyz, 2 * xxyy, 8 * xyyz, 4 * xyzz + 2 * xyyy, 4 * xyyz, 12 * yyyz, 8 * yyzz + 2 * yyyy, 4 * yzzz + 4 * yyyz, 6 * yyzz }, 
            { 0, 0, 2 * zz, 4 * yz, 0, 2 * xzz, 4 * xyz, 4 * yzz, 2 * zzz + 4 * yyz, 8 * yzz, 0, 2 * xxzz, 4 * xxyz, 4 * xyzz, 2 * xzzz + 4 * xyyz, 8 * xyzz, 6 * yyzz, 4 * yzzz + 4 * yyyz, 2 * zzzz + 8 * yyzz, 12 * yzzz }, 
            { 0, 0, 0, 6 * zz, 0, 0, 6 * xzz, 0, 6 * yzz, 12 * zzz, 0, 0, 6 * xxzz, 0, 6 * xyzz, 12 * xzzz, 0, 6 * yyzz, 12 * yzzz, 18 * zzzz },
        });
    }
}
