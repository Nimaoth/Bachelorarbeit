using System;
using System.Collections;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Single;
using Newtonsoft.Json.Converters;
using System.IO;
using Newtonsoft.Json;
using System.Threading;
using System.Runtime.InteropServices.WindowsRuntime;
using MathNet.Numerics.Random;
using UnityEditor;
using System.Dynamic;

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

    public vec3(Vector3 v)
    {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
    }
}

[Serializable]
public struct Sample {
    public float[] features;
    public float[] out_pos;
    public float absorbtion;
    //public vec3 entryPoint;
    //public vec3 exitPoint;
    //public float[] coefficients;
    //public float sigmaS;
    //public float sigmaA;
    //public float sigmaT;
    //public float g;
    //public float ior;
}

public struct AvgStats
{
    public double effectiveAlbedo;
    public double g;
    public double ior;
    public vec3 pos;
}

public class TestDataGenerator : MonoBehaviour
{
    private static MeshCollider meshCollider;

    private static Thread trainThread = null;
    private static System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
    private static TimeSpan elapsedTime;

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
            string outputPath,
            float distScale,
            Action<float> callback) {

        if (trainThread != null)
        {
            return;
        }

        if (meshCollider == null)
        {
            var tempObject = new GameObject();
            meshCollider = tempObject.AddComponent<MeshCollider>();
        }
        meshCollider.sharedMesh = mesh;

        // get random point on surface + normal
        Vector3[] meshVertices = mesh.vertices;
        Vector3[] meshNormals = mesh.normals;
        int[] meshTriangleIndices = mesh.triangles;

        // generate point cloud
        var random = new System.Random(randomVertexSeed);
        var pointCloud = ComputePointCloud(random, meshVertices, meshNormals, meshTriangleIndices, minimumPointCount, pointsPerAreaMultiplier, 0.1f);

        trainThread = new Thread(() =>
        {
            stopwatch.Restart();

            var samples = new List<Sample>(sampleCount);

            double effectiveAlbedoAvg = 0.0;
            double gAvg = 0.0;
            double iorAvg = 0.0;
            Vector3 posAvg = Vector3.zero;

            for (int i = 0; i < sampleCount; i++)
            {
                callback((float)i / sampleCount);

                // generate random point and normal on surface
                int randomTriangleIndex = (int)RandomRange(random, 0, meshTriangleIndices.Length / 3) * 3;
                var (b0, b1, b2) = GetRandomBarycentric(random);
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
                Vector3 randomNormalTemp = OnUnitSphere(random);
                randomNormalTemp.z = Mathf.Abs(randomNormalTemp.z);
                randomNormalTemp = randomNormalTemp.x * normalCSRight + randomNormalTemp.y * normalCSUp + randomNormalTemp.z * randomSurfacePointNormal;

                Vector3 direction = -randomNormalTemp.normalized;
                Vector3 origin = randomSurfacePoint - direction * 0.1f;

                // generate random surface properties
                float g = RandomRange(random, 0.1f, 0.95f);
                float ior = RandomRange(random, 1.0f, 1.5f);
                float sigmaS = RandomRange(random, 0.5f, 50.0f);
                float sigmaT = RandomRange(random, sigmaS + 0.1f, 51.0f);
                float sigmaA = sigmaT - sigmaS;
                float sigmaSRed = (1 - g) * sigmaS;
                float effectiveAlbedo = 0.0f;

                float standardDeviation = 0.0f;
                {
                    float sigmaTred = sigmaA + sigmaSRed;
                    float alphaRed = sigmaSRed / sigmaTred;

                    float e8 = Mathf.Pow((float)Math.E, 8);
                    effectiveAlbedo = 1 - (0.125f * Mathf.Log(e8 + alphaRed * (1 - e8)));

                    float mad = MAD(g, alphaRed);

                    standardDeviation = 2 * mad / sigmaTred;
                }

                effectiveAlbedoAvg += effectiveAlbedo;
                gAvg += g;
                iorAvg += ior;

                // calculate coefficients for random point on surfice
                float[] coefficients = TestDataGenerator.CalculateCoefficients(
                    random,
                    pointCloud,
                    randomSurfacePoint,
                    standardDeviation,
                    weightFactor,
                    weightScale,
                    maxVertexCount,
                    randomVertexCount,
                    randomVertexWeight,
                    largeCoefficientPenalty);


                //origin = new Vector3(0, 0, -1);
                //direction = new Vector3(0, 0, 1);

                int absorbtionSum = 0;
                int absorbtionTestCount = 128;

                Vector3? exitPointOpt = null;
                for (int k = 0; k < absorbtionTestCount; k++)
                {
                    var p = SimulatePath(random, coefficients, origin, direction, randomSurfacePoint, g, sigmaSRed, sigmaA, distScale, null);
                    if (exitPointOpt == null && p != null)
                        exitPointOpt = p;

                    if (p == null) absorbtionSum += 1;
                }

                float absorbtion = (float)absorbtionSum / (float)absorbtionTestCount;

                // @todo: handle absorbed data
                if (exitPointOpt != null)
                {
                    posAvg += exitPointOpt.Value;

                    // rotate polynomial so the incident direction aligns with the z-axis
                    Vector3 right = GetPerpendicular(direction);
                    Vector3 up = Vector3.Cross(right, direction);
                    RotatePolynomial(coefficients, right, up, direction);

                    Vector4 Vec3To4(Vector3 v, float w) => new Vector4(v.x, v.y, v.z, w);

                    // exit point should be in local space
                    var rotMatrix = new Matrix4x4(Vec3To4(right, 0), Vec3To4(up, 0), Vec3To4(-direction, 0), new Vector4(0, 0, 0, 1));
                    Vector3 exitPoint = rotMatrix * (exitPointOpt.Value - randomSurfacePoint);
                    samples.Add(new Sample
                    {
                        features = coefficients.Concat(new float[] { effectiveAlbedo, g, ior }).ToArray(),
                        out_pos = new float[] { exitPoint.x, exitPoint.y, exitPoint.z },
                        absorbtion = absorbtion,
                        //entryPoint      = new vec3(randomSurfacePoint.x, randomSurfacePoint.y, randomSurfacePoint.z),
                        //exitPoint       = new vec3(exitPoint.x, exitPoint.y, exitPoint.z),
                        //coefficients    = coefficients,
                        //sigmaS          = sigmaS,
                        //sigmaA          = sigmaA,
                        //sigmaT          = sigmaT,
                        //g               = g,
                        //ior             = ior
                    });
                }
            }
            callback(1);

            var jsonString = JsonConvert.SerializeObject(samples.ToArray(), Formatting.Indented);

            var now = DateTime.Now;
            File.WriteAllText($"../train_data/samples_{now.Year}-{now.Month}-{now.Day}--{now.Hour}-{now.Minute}-{now.Second}.json", jsonString);


            var avg = new AvgStats {
                effectiveAlbedo = effectiveAlbedoAvg / sampleCount,
                g = gAvg / sampleCount,
                ior = iorAvg / sampleCount,
                pos = new vec3(posAvg / sampleCount)
            };
            var avgJsonString = JsonConvert.SerializeObject(avg, Formatting.Indented);
            File.WriteAllText($"../train_data/stats_{now.Year}-{now.Month}-{now.Day}--{now.Hour}-{now.Minute}-{now.Second}.json", avgJsonString);

            elapsedTime = stopwatch.Elapsed;

            //Debug.Log($"Generating {sampleCount} samples took {time}");
            trainThread = null;
        });

        trainThread.Start();
    }


    private static float RandomRange(System.Random random, float min, float max) => (float)random.NextDouble() * (max - min) + min;
    
    private static Vector2 InsideUnitCircle(System.Random random)
    {
        while (true)
        {
            Vector2 vec = new Vector2((float)random.NextDouble() - 0.5f, (float)random.NextDouble() - 0.5f);
            if (vec.sqrMagnitude <= 1)
                return vec;
        }
    }
    private static Vector3 OnUnitSphere(System.Random random)
    {
        Vector3 vec = new Vector3((float)random.NextDouble() - 0.5f, (float)random.NextDouble() - 0.5f, (float)random.NextDouble() - 0.5f);
        return vec.normalized;
    }

    private static (float, float, float) GetRandomBarycentric(System.Random random)
    {
        float r1 = Mathf.Sqrt(RandomRange(random, 0.0f, 1.0f));
        float r2 = RandomRange(random, 0.0f, 1.0f);
        float a = 1 - r1;
        float b = r1 * (1 - r2);
        float c = r1 * r2;
        return (a, b, c);
    }


    private static void RotatePolynomial(float[] c, Vector3 right, Vector3 up, Vector3 forward)
    {
        float[] coefficientsTemp = new float[20];
        coefficientsTemp[0] = c[0];
        coefficientsTemp[1] = c[1] * right.x + c[2] * right.y + c[3] * right.z;
        coefficientsTemp[2] = c[1] * up.x + c[2] * up.y + c[3] * up.z;
        coefficientsTemp[3] = c[1] * forward.x + c[2] * forward.y + c[3] * forward.z;
        coefficientsTemp[4] = c[4] * Mathf.Pow(right.x, 2) + c[5] * right.x * right.y + c[6] * right.x * right.z + c[7] * Mathf.Pow(right.y, 2) + c[8] * right.y * right.z + c[9] * Mathf.Pow(right.z, 2);
        coefficientsTemp[5] = 2 * c[4] * right.x * up.x + c[5] * (right.x * up.y + right.y * up.x) + c[6] * (right.x * up.z + right.z * up.x) + 2 * c[7] * right.y * up.y + c[8] * (right.y * up.z + right.z * up.y) + 2 * c[9] * right.z * up.z;
        coefficientsTemp[6] = 2 * c[4] * forward.x * right.x + c[5] * (forward.x * right.y + forward.y * right.x) + c[6] * (forward.x * right.z + forward.z * right.x) + 2 * c[7] * forward.y * right.y + c[8] * (forward.y * right.z + forward.z * right.y) + 2 * c[9] * forward.z * right.z;
        coefficientsTemp[7] = c[4] * Mathf.Pow(up.x, 2) + c[5] * up.x * up.y + c[6] * up.x * up.z + c[7] * Mathf.Pow(up.y, 2) + c[8] * up.y * up.z + c[9] * Mathf.Pow(up.z, 2);
        coefficientsTemp[8] = 2 * c[4] * forward.x * up.x + c[5] * (forward.x * up.y + forward.y * up.x) + c[6] * (forward.x * up.z + forward.z * up.x) + 2 * c[7] * forward.y * up.y + c[8] * (forward.y * up.z + forward.z * up.y) + 2 * c[9] * forward.z * up.z;
        coefficientsTemp[9] = c[4] * Mathf.Pow(forward.x, 2) + c[5] * forward.x * forward.y + c[6] * forward.x * forward.z + c[7] * Mathf.Pow(forward.y, 2) + c[8] * forward.y * forward.z + c[9] * Mathf.Pow(forward.z, 2);
        coefficientsTemp[10] = c[10] * Mathf.Pow(right.x, 3) + c[11] * Mathf.Pow(right.x, 2) * right.y + c[12] * Mathf.Pow(right.x, 2) * right.z + c[13] * right.x * Mathf.Pow(right.y, 2) + c[14] * right.x * right.y * right.z + c[15] * right.x * Mathf.Pow(right.z, 2) + c[16] * Mathf.Pow(right.y, 3) + c[17] * Mathf.Pow(right.y, 2) * right.z + c[18] * right.y * Mathf.Pow(right.z, 2) + c[19] * Mathf.Pow(right.z, 3);
        coefficientsTemp[11] = 3 * c[10] * Mathf.Pow(right.x, 2) * up.x + c[11] * (Mathf.Pow(right.x, 2) * up.y + 2 * right.x * right.y * up.x) + c[12] * (Mathf.Pow(right.x, 2) * up.z + 2 * right.x * right.z * up.x) + c[13] * (2 * right.x * right.y * up.y + Mathf.Pow(right.y, 2) * up.x) + c[14] * (right.x * right.y * up.z + right.x * right.z * up.y + right.y * right.z * up.x) + c[15] * (2 * right.x * right.z * up.z + Mathf.Pow(right.z, 2) * up.x) + 3 * c[16] * Mathf.Pow(right.y, 2) * up.y + c[17] * (Mathf.Pow(right.y, 2) * up.z + 2 * right.y * right.z * up.y) + c[18] * (2 * right.y * right.z * up.z + Mathf.Pow(right.z, 2) * up.y) + 3 * c[19] * Mathf.Pow(right.z, 2) * up.z;
        coefficientsTemp[12] = 3 * c[10] * forward.x * Mathf.Pow(right.x, 2) + c[11] * (2 * forward.x * right.x * right.y + forward.y * Mathf.Pow(right.x, 2)) + c[12] * (2 * forward.x * right.x * right.z + forward.z * Mathf.Pow(right.x, 2)) + c[13] * (forward.x * Mathf.Pow(right.y, 2) + 2 * forward.y * right.x * right.y) + c[14] * (forward.x * right.y * right.z + forward.y * right.x * right.z + forward.z * right.x * right.y) + c[15] * (forward.x * Mathf.Pow(right.z, 2) + 2 * forward.z * right.x * right.z) + 3 * c[16] * forward.y * Mathf.Pow(right.y, 2) + c[17] * (2 * forward.y * right.y * right.z + forward.z * Mathf.Pow(right.y, 2)) + c[18] * (forward.y * Mathf.Pow(right.z, 2) + 2 * forward.z * right.y * right.z) + 3 * c[19] * forward.z * Mathf.Pow(right.z, 2);
        coefficientsTemp[13] = 3 * c[10] * right.x * Mathf.Pow(up.x, 2) + c[11] * (2 * right.x * up.x * up.y + right.y * Mathf.Pow(up.x, 2)) + c[12] * (2 * right.x * up.x * up.z + right.z * Mathf.Pow(up.x, 2)) + c[13] * (right.x * Mathf.Pow(up.y, 2) + 2 * right.y * up.x * up.y) + c[14] * (right.x * up.y * up.z + right.y * up.x * up.z + right.z * up.x * up.y) + c[15] * (right.x * Mathf.Pow(up.z, 2) + 2 * right.z * up.x * up.z) + 3 * c[16] * right.y * Mathf.Pow(up.y, 2) + c[17] * (2 * right.y * up.y * up.z + right.z * Mathf.Pow(up.y, 2)) + c[18] * (right.y * Mathf.Pow(up.z, 2) + 2 * right.z * up.y * up.z) + 3 * c[19] * right.z * Mathf.Pow(up.z, 2);
        coefficientsTemp[14] = 6 * c[10] * forward.x * right.x * up.x + c[11] * (2 * forward.x * right.x * up.y + 2 * forward.x * right.y * up.x + 2 * forward.y * right.x * up.x) + c[12] * (2 * forward.x * right.x * up.z + 2 * forward.x * right.z * up.x + 2 * forward.z * right.x * up.x) + c[13] * (2 * forward.x * right.y * up.y + 2 * forward.y * right.x * up.y + 2 * forward.y * right.y * up.x) + c[14] * (forward.x * right.y * up.z + forward.x * right.z * up.y + forward.y * right.x * up.z + forward.y * right.z * up.x + forward.z * right.x * up.y + forward.z * right.y * up.x) + c[15] * (2 * forward.x * right.z * up.z + 2 * forward.z * right.x * up.z + 2 * forward.z * right.z * up.x) + 6 * c[16] * forward.y * right.y * up.y + c[17] * (2 * forward.y * right.y * up.z + 2 * forward.y * right.z * up.y + 2 * forward.z * right.y * up.y) + c[18] * (2 * forward.y * right.z * up.z + 2 * forward.z * right.y * up.z + 2 * forward.z * right.z * up.y) + 6 * c[19] * forward.z * right.z * up.z;
        coefficientsTemp[15] = 3 * c[10] * Mathf.Pow(forward.x, 2) * right.x + c[11] * (Mathf.Pow(forward.x, 2) * right.y + 2 * forward.x * forward.y * right.x) + c[12] * (Mathf.Pow(forward.x, 2) * right.z + 2 * forward.x * forward.z * right.x) + c[13] * (2 * forward.x * forward.y * right.y + Mathf.Pow(forward.y, 2) * right.x) + c[14] * (forward.x * forward.y * right.z + forward.x * forward.z * right.y + forward.y * forward.z * right.x) + c[15] * (2 * forward.x * forward.z * right.z + Mathf.Pow(forward.z, 2) * right.x) + 3 * c[16] * Mathf.Pow(forward.y, 2) * right.y + c[17] * (Mathf.Pow(forward.y, 2) * right.z + 2 * forward.y * forward.z * right.y) + c[18] * (2 * forward.y * forward.z * right.z + Mathf.Pow(forward.z, 2) * right.y) + 3 * c[19] * Mathf.Pow(forward.z, 2) * right.z;
        coefficientsTemp[16] = c[10] * Mathf.Pow(up.x, 3) + c[11] * Mathf.Pow(up.x, 2) * up.y + c[12] * Mathf.Pow(up.x, 2) * up.z + c[13] * up.x * Mathf.Pow(up.y, 2) + c[14] * up.x * up.y * up.z + c[15] * up.x * Mathf.Pow(up.z, 2) + c[16] * Mathf.Pow(up.y, 3) + c[17] * Mathf.Pow(up.y, 2) * up.z + c[18] * up.y * Mathf.Pow(up.z, 2) + c[19] * Mathf.Pow(up.z, 3);
        coefficientsTemp[17] = 3 * c[10] * forward.x * Mathf.Pow(up.x, 2) + c[11] * (2 * forward.x * up.x * up.y + forward.y * Mathf.Pow(up.x, 2)) + c[12] * (2 * forward.x * up.x * up.z + forward.z * Mathf.Pow(up.x, 2)) + c[13] * (forward.x * Mathf.Pow(up.y, 2) + 2 * forward.y * up.x * up.y) + c[14] * (forward.x * up.y * up.z + forward.y * up.x * up.z + forward.z * up.x * up.y) + c[15] * (forward.x * Mathf.Pow(up.z, 2) + 2 * forward.z * up.x * up.z) + 3 * c[16] * forward.y * Mathf.Pow(up.y, 2) + c[17] * (2 * forward.y * up.y * up.z + forward.z * Mathf.Pow(up.y, 2)) + c[18] * (forward.y * Mathf.Pow(up.z, 2) + 2 * forward.z * up.y * up.z) + 3 * c[19] * forward.z * Mathf.Pow(up.z, 2);
        coefficientsTemp[18] = 3 * c[10] * Mathf.Pow(forward.x, 2) * up.x + c[11] * (Mathf.Pow(forward.x, 2) * up.y + 2 * forward.x * forward.y * up.x) + c[12] * (Mathf.Pow(forward.x, 2) * up.z + 2 * forward.x * forward.z * up.x) + c[13] * (2 * forward.x * forward.y * up.y + Mathf.Pow(forward.y, 2) * up.x) + c[14] * (forward.x * forward.y * up.z + forward.x * forward.z * up.y + forward.y * forward.z * up.x) + c[15] * (2 * forward.x * forward.z * up.z + Mathf.Pow(forward.z, 2) * up.x) + 3 * c[16] * Mathf.Pow(forward.y, 2) * up.y + c[17] * (Mathf.Pow(forward.y, 2) * up.z + 2 * forward.y * forward.z * up.y) + c[18] * (2 * forward.y * forward.z * up.z + Mathf.Pow(forward.z, 2) * up.y) + 3 * c[19] * Mathf.Pow(forward.z, 2) * up.z;
        coefficientsTemp[19] = c[10] * Mathf.Pow(forward.x, 3) + c[11] * Mathf.Pow(forward.x, 2) * forward.y + c[12] * Mathf.Pow(forward.x, 2) * forward.z + c[13] * forward.x * Mathf.Pow(forward.y, 2) + c[14] * forward.x * forward.y * forward.z + c[15] * forward.x * Mathf.Pow(forward.z, 2) + c[16] * Mathf.Pow(forward.y, 3) + c[17] * Mathf.Pow(forward.y, 2) * forward.z + c[18] * forward.y * Mathf.Pow(forward.z, 2) + c[19] * Mathf.Pow(forward.z, 3);
        for (int i = 0; i < coefficientsTemp.Length; ++i)
        {
            c[i] = coefficientsTemp[i];
        }
    }

    private static float MAD(float g, float a_red) {
        float e8 = Mathf.Pow((float)Math.E, 8);
        float a_eff = 1 - (0.125f * Mathf.Log(e8 + a_red * (1 - e8)));
        return 0.25f * g + 0.25f * a_red + a_eff;
    }

    public static Vector3? SimulatePath(System.Random random, float[] coefficients, Vector3 origin, Vector3 direction, Vector3 center, float g, float sigmaSRed, float sigmaA, float distScale, Action<PathPoint> onInteraction) {
        direction = direction.normalized;

        float maxDist = float.MaxValue;

        float throughput = 1.0f;

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
                // exited surface
                return newPoint;
            }

            // still inside, set random direction and maxDist
            if (g == 0) {
                direction = OnUnitSphere(random);
            } else {
                float rand = RandomRange(random, 0.0f, 1.0f);
                float cosAngle = InvIntPhaseFunction(rand, g);
                float theta = Mathf.Acos(cosAngle);
                float d = 1 / Mathf.Tan(theta);

                // Debug.Log($"rand: {rand}, cosAngle: {cosAngle}, theta: {theta}, d: {d}");

                if (d == float.PositiveInfinity) {
                    // do nothing
                    //direction = direction;
                } else if (d == float.NegativeInfinity) {
                    direction = -direction;
                } else {
                    var right = GetPerpendicular(direction).normalized;
                    var up = Vector3.Cross(right, direction).normalized;
                    var uv = InsideUnitCircle(random);

                    point.uv = uv;
                    point.right = right;
                    point.up = up;

                    // Debug.Log($"right: {right}, up: {up}, uv: {uv}");

                    direction = uv.x * right + uv.y * up + d * direction;
                    direction = direction.normalized;
                }
            }

            // Debug.Log($"direction: {direction}");

            maxDist = GetScatterDistance(random, sigmaSRed) * distScale;
            point.direction = point.direction * Vector3.Dot(point.direction, direction * maxDist);

            onInteraction?.Invoke(point);

            float transmission = Mathf.Exp(-sigmaA * maxDist);

            throughput = throughput * transmission;

            // russian roulette
            if (i > 5)
            {
                if ((float)random.NextDouble() > throughput)
                    return null;
                throughput = 1;
            }
        }

        // reached max steps, ray counts as absorbed
        return null;
    }

    private static float GetScatterDistance(System.Random random, float sigmaSRed) {
        float dist = -Mathf.Log((float)random.NextDouble()) / sigmaSRed;
        return dist;
    }

    private static float InvIntPhaseFunction(float x, float g) {
        float g2 = g * g;

        float tmp = (1 - g2) / (1 + g * (2 * x - 1));
        return (1 / (2 * g)) * (1 + g2 - tmp * tmp);
    }

    public static Vertex[] ComputePointCloud(System.Random random, Vector3[] positions, Vector3[] normals, int[] triangles, int minimumPointCount, float pointsPerAreaMultiplier, float standardDeviation) {
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

            if (random.NextDouble() < (pointsInTriangle - (int)pointsInTriangle)) {
                pointsInTriangle += 1;
            }
            // Debug.Log($"Tri {i / 3}: area = {area} ({area/totalSurfaceArea*100}% of total), generating {(int)pointsInTriangle} points");

            for (int k = 0; k < (int)pointsInTriangle; k++) {
                var (a, b, c) = GetRandomBarycentric(random);

                Vector3 pos = (a * v1 + b * v2 + c * v3);
                Vector3 normal = (a * n1 + b * n2 + c * n3).normalized;
                vertices.Add(new Vertex(pos, normal));
            }
        }

        // Debug.Log($"Generated {vertices.Count} points");
        return vertices.ToArray();
    }

    public static float[] CalculateCoefficients(System.Random random, Vertex[] vertices, Vector3 center, float standardDeviation, float weightFactor, float weightScale, int maxVertexCount, int randomVertexCount, float randomVertexWeight, float largeCoefficientPenalty) {
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
