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
    private int pointsPerArea = 100;

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
    private LineRenderer pathRenderer;


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
                pointsPerArea = min;
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

    private void SimulatePath(Vector3 origin, Vector3 direction) {
        var points = new List<Vector3>();
        points.Add(origin);

        float maxDist = float.MaxValue;

        for (int i = 0; i < 1500; i++) {
            var (dist, inside) = surface.RayMarch(origin, direction, maxDist);
            var newPoint = origin + direction * dist;
            points.Add(newPoint);

            var normal = surface.GetNormalAt(newPoint);
            origin = newPoint - normal * surface.SURF_DIST * 2;

            if (!inside) {
                // @todo: refract
                var (_dist, _) = surface.RayMarch(origin, direction, 10.0f);
                points.Add(origin + direction * _dist);
                break;
            }

            // still inside, set random direction and maxDist
            if (surface.g == 0) {
                direction = UnityEngine.Random.onUnitSphere;
            } else {
                float rand = UnityEngine.Random.Range(0.0f, 1.0f);
                float cosAngle = InvIntPhaseFunction(rand, surface.g);
                float theta = Mathf.Acos(cosAngle);
                float d = 1 / Mathf.Tan(theta);

                // Debug.Log($"rand: {rand}, cosAngle: {cosAngle}, theta: {theta}, d: {d}");

                if (d == float.PositiveInfinity) {

                } else if (d == float.NegativeInfinity) {
                    direction = -direction;
                } else {
                    var right = GetPerpendicular(direction).normalized;
                    var up = Vector3.Cross(right, direction);
                    var uv = UnityEngine.Random.insideUnitCircle.normalized;

                    // Debug.Log($"right: {right}, up: {up}, uv: {uv}");

                    direction = uv.x * right + uv.y * up + d * direction * Mathf.Sign(cosAngle);
                    direction = direction.normalized;
                }
            }

            // Debug.Log($"direction: {direction}");

            // direction = UnityEngine.Random.onUnitSphere;
            maxDist = UnityEngine.Random.Range(0.001f, 0.01f);
            maxDist = 0.05f;
        }

        pathRenderer.positionCount = points.Count;
        pathRenderer.SetPositions(points.ToArray());
    }

    #region Point Cloud

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

    private float AreaOfTriangle(Vector3 v1, Vector3 v2, Vector3 v3) {
        var a = v2 - v1;
        var b = v3 - v1;
        return Vector3.Cross(a, b).magnitude * 0.5f;
    }

    private void ComputePoints(PointCloud pointCloud) {
        Debug.Log($"minimumPointCount: {minimumPointCount}");
        Debug.Log($"pointsPerArea: {pointsPerArea}");
        Mesh mesh = pointCloud.mesh;
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

        int pointCount = Mathf.Max(minimumPointCount, (int)totalSurfaceArea * pointsPerArea);
        // pointCount = Mathf.Max(pointCount, triangles.Length);

        // Debug.Log($"Mesh {mesh.name}: area = {totalSurfaceArea}, generating {pointCount} points");

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
                float r1 = Mathf.Sqrt(UnityEngine.Random.Range(0.0f, 1.0f));
                float r2 = UnityEngine.Random.Range(0.0f, 1.0f);
                float a = 1 - r1;
                float b = r1 * (1 - r2);
                float c = r1 * r2;

                Vector3 pos = (a * v1 + b * v2 + c * v3);
                Vector3 normal = (a * n1 + b * n2 + c * n3).normalized;
                vertices.Add(new Vertex(pos, normal));
            }
        }

        // Debug.Log($"Generated {vertices.Count} points");

        pointCloud.vertices = vertices.ToArray();
    }

    private void SetActivePointCloud(PointCloud pc) {

        currentPointCloud = pc;

        if (currentPointCloud.vertices.Length <= pointCloudBuffer.count)
            pointCloudBuffer.SetData(currentPointCloud.vertices);
        else
            Debug.LogError("Too many points in point cloud: {currentPointCloud.vertices.Length}/{pointCloudBuffer.count}");
    }

    #endregion

    #region Poly Surface

    private void RecalculateCoefficientsThread() {
        int targetFPS = 120;
        int msPerFrame = 1000 / targetFPS;

        var watch = new System.Diagnostics.Stopwatch();
        var random = new System.Random();

        while (true) {
            watch.Restart();
            if (currentVertices?.Length > 0) {
                random = new System.Random(randomVertexSeed);

                var center = currentCenter;
                float maxDist = -2 * (Mathf.Log(0.01f)) / weightScale;
                maxDist = maxDist * maxDist;
                Vertex[] vertices = currentVertices;
                vertices = vertices
                    .Where(v => (v.position - center).sqrMagnitude <= maxDist)
                    .OrderBy(v => (v.position - center).sqrMagnitude)
                    .Take(Mathf.Min(vertices.Length, maxVertexCount))
                    .Concat(Enumerable.Repeat(0, randomVertexCount).Select(_ => vertices[random.Next() % vertices.Length].withWeight(randomVertexWeight)))
                    .Select(v => new Vertex(v.position - center, v.normal, v.weight))
                    .ToArray();
                currentClosestPoints = vertices;


                var X = DenseMatrix.OfRows(vertices.Length, 20, vertices.Select(v => Pc(v)));
                var tmp = 2 * X.Transpose() * X + Sum(20, 20, vertices, v => weight(v) * Y(v.position)) + 2 * largeCoefficientPenalty;
                var c = tmp.Inverse() * Sum(20, 1, vertices, v => weight(v) * Z(v.position, v.normal));

                coefficients = c.Column(0).ToArray();
                coefficients[0] = 0;
                surface.SetCenter(center);
                surface.SetCoefficients(coefficients);
            }
            var time = watch.Elapsed;
            if (time.Milliseconds < msPerFrame)
                Thread.Sleep(msPerFrame - time.Milliseconds);
        }
    }

    private DenseMatrix Sum(int r, int c, Vertex[] vertices, Func<Vertex, DenseMatrix> func) {
        DenseMatrix result = new DenseMatrix(r, c);
        for (int i = 0; i < vertices.Length; i++) {
            result += func(vertices[i]);
        }
        return result;
    }

    private float weight(Vertex v) {
        float weight = v.weight;
        if (weight < 0)
            weight = Mathf.Exp(-0.5f * v.position.sqrMagnitude * weightScale);
        return Mathf.Lerp(1, weight, weightFactor);
    }

    private float[] Pc(Vertex v) {
        float w = Mathf.Sqrt(weight(v));
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

    private DenseMatrix Z(Vector3 p, Vector3 n) {
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

    private DenseMatrix Y(Vector3 p) {
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

    #endregion
}
