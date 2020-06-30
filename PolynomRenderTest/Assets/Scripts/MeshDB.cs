using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra.Complex.Solvers;
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
    [Range(0, 1)]
    private float distScale = 1.0f;

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
            //SimulatePath(point.origin, point.direction);

            var random = new System.Random();

            currentPath.Clear();
            currentPath.Add(new PathPoint
            {
                position = point.origin,
                direction = point.direction
            });

            var exit = TestDataGenerator.SimulatePath(
                random,
                surface.Coefficients,
                point.origin, point.direction,
                surface.Center,
                surface.g, surface.SigmaSReduced, surface.sigmaA,
                distScale,
                (p) => currentPath.Add(p));

            if (exit != null)
            {
                currentPath.Add(new PathPoint
                {
                    position = exit.Value,
                    direction = (exit.Value - currentPath.Last().position).normalized,
                });

                currentPath.Add(new PathPoint
                {
                    position = currentPath.Last().position + currentPath.Last().direction * 10,
                });
            }
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

    public void GenerateMeshWithCoefficients()
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

        var mesh = currentPointCloud.mesh;
        var vertices = mesh.vertices;
        var uvs = mesh.uv;
        var normals = mesh.normals;
        var indices = mesh.triangles;

        var builder = new StringBuilder();
        builder.AppendLine("o " + mesh.name);

        foreach (var v in vertices)
            builder.AppendLine($"v {v.x} {v.y} {v.z}");
        foreach (var v in normals)
            builder.AppendLine($"vn {v.x} {v.y} {v.z}");
        foreach (var v in uvs)
            builder.AppendLine($"vt {v.x} {v.y}");

        var random = new System.Random();
        foreach (var v in vertices)
        {
            // calculate coefficients for random point on surfice
            float[] coefficients = TestDataGenerator.CalculateCoefficients(
                random,
                currentPointCloud.vertices,
                v,
                1,
                weightFactor,
                weightScale,
                maxVertexCount,
                randomVertexCount,
                randomVertexWeight,
                largeCoefficientPenalty);

            builder.Append($"vc");
            foreach (float c in coefficients)
                builder.Append(" " + c);
            builder.AppendLine();
        }

        string format = "f {0}/{0}/{0}/{0} {1}/{1}/{1}/{1} {2}/{2}/{2}/{2}\n";
        if (uvs.Length == 0)
            format = "f {0}//{0}/{0} {1}//{1}/{1} {2}//{2}/{2}\n";
        //string format = "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n";
        //if (uvs.Length == 0)
        //    format = "f {0}//{0} {1}//{1} {2}//{2}\n";

        for (int i = 0; i < indices.Length; i += 3)
        {
            int i0 = indices[i] + 1;
            int i1 = indices[i + 1] + 1;
            int i2 = indices[i + 2] + 1;

            builder.AppendFormat(format, i0, i1, i2);
        }

        File.WriteAllText(mesh.name + ".obj", builder.ToString());
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
            distScale,
            progress => {
                trainingProgress = progress;
            });
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
