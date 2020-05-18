using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using MathNet.Numerics.LinearAlgebra.Single;
using UnityEngine;
using UnityEngine.UI;

public struct Vertex {
    public const int SIZE_IN_BYTES = sizeof(float) * (3 + 3 + 1);
    public Vector3 position;
    public Vector3 normal;
    public float weight;

    public Vertex(Vector3 pos, Vector3 normal, float weight = -1.0f)
    {
        this.position = pos;
        this.normal = normal;
        this.weight = weight;
    }

    public Vertex withWeight(float w) {
        return new Vertex(position, normal) {
            weight = w
        };
    }
}

public class PointCloud {
    public Mesh mesh;
    public Vertex[] vertices;

    public PointCloud(Mesh mesh) {
        this.mesh = mesh;
    }
}

public struct PathPoint {
    public Vector3 position;
    public Vector3? uv;
    public Vector3? right;
    public Vector3? up;
    public Vector3 direction;
}

public class MeshDB : MonoBehaviour
{
    [SerializeField]
    private new Camera camera;

    [SerializeField]
    private Material pointCloudMaterial;

    [SerializeField]
    private Material meshMaterial;

    [SerializeField]
    private GameObject meshSelectorButtonList;

    [SerializeField]
    private GameObject meshSelectorButtonPrefab;

    // min point count
    [SerializeField]
    private InputField ifMinimumPointCount;
    private int minimumPointCount = 1024;

    // points per area
    [SerializeField]
    private InputField ifPointsPerArea;
    private float pointsPerAreaMultiplier = 1;

    // show point cloud
    [SerializeField]
    private Toggle cbShowPointCloud;
    private bool showPointCloud = true;

    // show mesh
    [SerializeField]
    private Toggle cbShowMesh;
    private bool showMesh = true;

    // 

    [SerializeField]
    private Transform hitLocation;

    [SerializeField]
    private PolySurface surface;
    private MeshCollider meshCollider;

    [SerializeField]
    [Min(0)]
    private float largeCoefficientPenalty = 0.0001f;

    [SerializeField]
    [Min(0)]
    private int maxVertexCount;

    [SerializeField]
    private int randomVertexSeed = 0;

    [SerializeField]
    [Min(0)]
    private int randomVertexCount;

    [SerializeField]
    [Range(0, 1)]
    private float randomVertexWeight;

    [SerializeField]
    [Range(1.0f, 100.0f)]
    private float weightScale = 1.0f;

    [SerializeField]
    [Range(0, 1)]
    private float weightFactor = 1.0f;

    [SerializeField]
    private Mesh[] objects;


    // private stuff
    private Thread thread;
    private Vertex[] currentVertices;
    private Vertex[] currentClosestPoints;
    private Vector3 currentCenter;
    private float[] coefficients;

    // point cloud visualization
    private PointCloud[] pointClouds;
    private PointCloud currentPointCloud;
    private ComputeBuffer pointCloudBuffer;

    // path
    [SerializeField]
    public float PathMinDist = 0.01f;
    public float PathMaxDist = 0.1f;
    public float PathAxisLen = 0.1f;
    public float Temp = 100;
    public bool RenderPathAxis = false;
    public bool RenderPath = true;
    private List<PathPoint> currentPath = new List<PathPoint>();

    public float pathStartRadius = 5.0f;

    public int dataSetSize = 100;

    [Range(0, 1)]
    public float trainingProgress = 0.0f;

    // methods

    void Start() {
        meshCollider = GetComponent<MeshCollider>();
        pointClouds = new PointCloud[objects.Length];

        for (int i = 0; i < objects.Length; i++) {
            var mesh = objects[i];
            var pointCloud = pointClouds[i] = new PointCloud(mesh);

            var button = GameObject.Instantiate(meshSelectorButtonPrefab, meshSelectorButtonList.transform);
            button.GetComponentInChildren<Text>().text = mesh.name;
            button.GetComponent<Button>().onClick.AddListener(() => {
                if (pointCloud.vertices == null) {
                    ComputePoints(pointCloud);
                }
                SetActivePointCloud(pointCloud);
            });
        }

        cbShowPointCloud.onValueChanged.AddListener(val => {
            showPointCloud = val;
        });

        cbShowMesh.onValueChanged.AddListener(val => {
            showMesh = val;
        });

        ifMinimumPointCount.onEndEdit.AddListener(val => {
            if (int.TryParse(val, out var min))
                minimumPointCount = min;
            else
                ifMinimumPointCount.text = "0";
            ClearAllPointClouds();
            RecomputeCurrentPointCloud();
        });

        ifPointsPerArea.onEndEdit.AddListener(val => {
            if (int.TryParse(val, out var min))
                pointsPerAreaMultiplier = min;
            else
                ifPointsPerArea.text = "0";
            ClearAllPointClouds();
            RecomputeCurrentPointCloud();
        });

        pointCloudBuffer = new ComputeBuffer(10000000, Vertex.SIZE_IN_BYTES);
        pointCloudMaterial.SetBuffer("pointCloud", pointCloudBuffer);

        ComputePoints(pointClouds[0]);
        SetActivePointCloud(pointClouds[0]);

        thread = new Thread(RecalculateCoefficientsThread);
        thread.Start();
    }

    void OnApplicationQuit() {
        pointCloudBuffer.Release();
    }

    private void OnRenderObject() {
        if (currentPointCloud == null)
            return;

        if (showMesh) {
            meshMaterial.SetPass(0);
            Graphics.DrawMeshNow(currentPointCloud.mesh, Matrix4x4.identity);
            // Graphics.DrawMesh(currentPointCloud.mesh, Matrix4x4.identity, meshMaterial, 0);
        }

        if (showPointCloud) {
            pointCloudMaterial.SetBuffer("pointCloud", pointCloudBuffer);
            pointCloudMaterial.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Points, 1, currentPointCloud.vertices.Length);
        }
    }

    private void Update() {
        var ctrl = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);
            
        pointCloudMaterial.SetFloat("_WeightScale", weightScale);
        pointCloudMaterial.SetFloat("_WeightFactor", weightFactor);

        if (currentPointCloud != null) {
            if (ctrl && Input.GetMouseButtonDown(1)) {
                var ray = camera.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out var hit)) {
                    camera.transform.parent.position = hit.point;
                } else {
                    camera.transform.parent.position = Vector3.zero;
                }
            }
            if (!ctrl && Input.GetMouseButton(1)) {
                var ray = camera.ScreenPointToRay(Input.mousePosition);
                meshCollider.sharedMesh = currentPointCloud.mesh;
                if (meshCollider.Raycast(ray, out var hit, float.MaxValue)) {
                    hitLocation.position = hit.point;
                    currentCenter = hit.point;
                    currentVertices = currentPointCloud.vertices;
                    pointCloudMaterial.SetVector("center", hit.point);
                }
            }
        }

        if (ctrl && Input.GetMouseButtonDown(0)) {
            var point = camera.ScreenPointToRay(Input.mousePosition);
            SimulatePath(point.origin, point.direction);


        }

        for (int i = 0; i < currentPath.Count; i++) {
            var p = currentPath[i];

            if (RenderPathAxis) {
                if (p.uv != null) {
                    var vec = p.right.Value * p.uv.Value.x + p.up.Value * p.uv.Value.y;
                    Debug.DrawRay(p.position, vec * PathAxisLen, Color.yellow, 0, false);
                }
                if (p.right != null)
                    Debug.DrawRay(p.position, p.right.Value * PathAxisLen, Color.red, 0, false);
                if (p.up != null)
                    Debug.DrawRay(p.position, p.up.Value * PathAxisLen, Color.green, 0, false);
                Debug.DrawRay(p.position, p.direction, Color.blue, 0, false);
            }

            if (RenderPath && i < currentPath.Count - 1) {
                var col = Color.white;
                if (i == 0) {
                    col = Color.cyan;
                } else if (i == currentPath.Count - 2) {
                    col = Color.magenta;
                }
                Debug.DrawLine(p.position, currentPath[i + 1].position, col, 0, false);
            }
        }
    }

    private float InvIntPhaseFunction(float x, float g) {
        float g2 = g * g;

        float tmp = (1 - g2) / (1 + g * (2 * x - 1));
        return (1 / (2 * g)) * (1 + g2 - tmp * tmp);
    }

    private Vector3 GetPerpendicular(Vector3 p) {
        var t = new Vector3(p.z, p.z, -p.x - p.y);
        if (t.sqrMagnitude < 0.01f) {
            return new Vector3(-p.y - p.z, p.x, p.x);
        } else {
            return t;
        }
    }

    public void SimulatePathRandomPath() {
        TestDataGenerator.GenerateTestData(
            currentPointCloud.mesh,
            minimumPointCount,
            pointsPerAreaMultiplier,
            weightFactor,
            weightScale,
            maxVertexCount,
            randomVertexCount,
            randomVertexWeight,
            largeCoefficientPenalty,
            randomVertexSeed,
            dataSetSize,
            "",
            progress => {
                trainingProgress = progress;
            });

        // get random point on surface + normal
        // Vector3[] meshVertices = currentPointCloud.mesh.vertices;
        // Vector3[] meshNormals = currentPointCloud.mesh.normals;
        // int[] meshTriangleIndices = currentPointCloud.mesh.triangles;
        // int randomTriangleIndex = UnityEngine.Random.Range(0, meshTriangleIndices.Length / 3) * 3;
        
        // var (b0, b1, b2) = TestDataGenerator.GetRandomBarycentric();
        // Vector3 randomSurfacePoint =
        //     b0 * meshVertices[meshTriangleIndices[randomTriangleIndex + 0]] +
        //     b1 * meshVertices[meshTriangleIndices[randomTriangleIndex + 1]] +
        //     b2 * meshVertices[meshTriangleIndices[randomTriangleIndex + 2]];
        // Vector3 randomSurfacePointNormal =
        //     b0 * meshNormals[meshTriangleIndices[randomTriangleIndex + 0]] +
        //     b1 * meshNormals[meshTriangleIndices[randomTriangleIndex + 1]] +
        //     b2 * meshNormals[meshTriangleIndices[randomTriangleIndex + 2]];
        // randomSurfacePointNormal = randomSurfacePointNormal.normalized;

        // Vector3 normalCSRight = GetPerpendicular(randomSurfacePointNormal);
        // Vector3 normalCSUp = Vector3.Cross(randomSurfacePointNormal, normalCSRight);
        // Vector3 randomNormalTemp = UnityEngine.Random.onUnitSphere;
        // randomNormalTemp.z = Mathf.Abs(randomNormalTemp.z);
        // randomNormalTemp = randomNormalTemp.x * normalCSRight + randomNormalTemp.y * normalCSUp + randomNormalTemp.z * randomSurfacePointNormal;

        // Vector3 direction = -randomNormalTemp.normalized;
        // Vector3 origin = randomSurfacePoint - direction * pathStartRadius;
        // Debug.DrawLine(new Vector3(-pathStartRadius, 0, 0) + randomSurfacePoint, new Vector3(pathStartRadius, 0, 0) + randomSurfacePoint, Color.white, 1.5f);
        // Debug.DrawLine(new Vector3(0, -pathStartRadius, 0) + randomSurfacePoint, new Vector3(0, pathStartRadius, 0) + randomSurfacePoint, Color.white, 1.5f);
        // Debug.DrawLine(new Vector3(0, 0, -pathStartRadius) + randomSurfacePoint, new Vector3(0, 0, pathStartRadius) + randomSurfacePoint, Color.white, 1.5f);
        // Debug.DrawLine(randomSurfacePoint, randomSurfacePoint + randomSurfacePointNormal * 10, Color.red, 1.5f);
        // // Debug.DrawLine(randomSurfacePoint, randomSurfacePoint + direction * 10, Color.green, 1.5f);

        // camera.transform.parent.position = randomSurfacePoint;

        // meshCollider.sharedMesh = currentPointCloud.mesh;
        // if (meshCollider.Raycast(new Ray(origin, direction), out var hit, float.MaxValue)) {
        //     hitLocation.position = hit.point;
        //     currentCenter = hit.point;
        //     pointCloudMaterial.SetVector("center", hit.point);

        //     coefficients = TestDataGenerator.CalculateCoefficients(
        //         currentVertices,
        //         currentCenter,
        //         surface.StandardDeviation,
        //         weightFactor,
        //         weightScale,
        //         maxVertexCount,
        //         randomVertexCount,
        //         randomVertexWeight,
        //         largeCoefficientPenalty,
        //         randomVertexSeed);
        //     surface.SetCenter(currentCenter);
        //     surface.SetCoefficients(coefficients);

        //     SimulatePath(origin, direction);
        // }
    }

    private void SimulatePath(Vector3 origin, Vector3 direction) {
        currentPath.Clear();

        direction = direction.normalized;

        currentPath.Add(new PathPoint{
            position = origin,
            direction = direction
        });

        float maxDist = float.MaxValue;

        for (int i = 0; i < 1500; i++) {
            var (dist, inside) = PolySurface.RayMarch(surface.Coefficients, origin, direction, maxDist, surface.Center);
            var newPoint = origin + direction * dist;
            var point = new PathPoint{
                position = newPoint,
                direction = direction
            };

            var normal = PolySurface.GetNormalAt(surface.Coefficients, newPoint, surface.Center);
            origin = newPoint - normal * PolySurface.SURF_DIST * 2;

            if (!inside) {
                // @todo: refract
                var (_dist, _) = PolySurface.RayMarch(surface.Coefficients, origin, direction, 10.0f, surface.Center);
                currentPath.Add(point);
                currentPath.Add(new PathPoint{
                    position = origin + direction * 10,
                });
                break;
            }

            var g = surface.g;
            // g = Mathf.Clamp(Mathf.Lerp(-1, 1, surface.EvaluateAt(origin) / Temp + 1), -1, 1);

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
            // maxDist = UnityEngine.Random.Range(PathMinDist, PathMaxDist);
            maxDist = -Mathf.Log(UnityEngine.Random.Range(0.001f, 1)) / surface.sigmaS;
            point.direction = point.direction * Vector3.Dot(point.direction, direction * maxDist);
            currentPath.Add(point);
        }
    }

    private void RecomputeCurrentPointCloud() {
        if (currentPointCloud == null)
            return;
        ComputePoints(currentPointCloud);
        SetActivePointCloud(currentPointCloud);
    }

    private void ClearAllPointClouds() {
        foreach (var p in pointClouds) {
            p.vertices = null;
        }
    }

    private void ComputePoints(PointCloud pointCloud) {
        var random = new System.Random(randomVertexSeed);
        pointCloud.vertices = TestDataGenerator.ComputePointCloud(random, pointCloud.mesh.vertices, pointCloud.mesh.normals, pointCloud.mesh.triangles, minimumPointCount, pointsPerAreaMultiplier, surface.StandardDeviation);
    }

    private void SetActivePointCloud(PointCloud pc) {
        currentPointCloud = pc;

        if (currentPointCloud.vertices.Length <= pointCloudBuffer.count)
            pointCloudBuffer.SetData(currentPointCloud.vertices);
        else
            Debug.LogError("Too many points in point cloud: {currentPointCloud.vertices.Length}/{pointCloudBuffer.count}");
    }

    private void RecalculateCoefficientsThread() {
        int targetFPS = 120;
        int msPerFrame = 1000 / targetFPS;

        var watch = new System.Diagnostics.Stopwatch();
        var random = new System.Random(randomVertexSeed);

        while (true) {
            watch.Restart();
            if (currentVertices?.Length > 0) {
                coefficients = TestDataGenerator.CalculateCoefficients(
                    random,
                    currentVertices,
                    currentCenter,
                    surface.StandardDeviation,
                    weightFactor,
                    weightScale,
                    maxVertexCount,
                    randomVertexCount,
                    randomVertexWeight,
                    largeCoefficientPenalty);
                surface.SetCenter(currentCenter);
                surface.SetCoefficients(coefficients);
            }
            var time = watch.Elapsed;
            if (time.Milliseconds < msPerFrame)
                Thread.Sleep(msPerFrame - time.Milliseconds);
        }
    }
}
