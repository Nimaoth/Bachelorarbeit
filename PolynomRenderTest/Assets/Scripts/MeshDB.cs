using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public struct Vertex {
    public const int SIZE_IN_BYTES = sizeof(float) * 3 * 2;
    public Vector3 position;
    public Vector3 normal;

    public Vertex(Vector3 pos, Vector3 normal)
    {
        this.position = pos;
        this.normal = normal;
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
    private Mesh[] objects;

    private PointCloud[] pointClouds;


    // point cloud visualization
    private PointCloud currentPointCloud;
    private ComputeBuffer pointCloudBuffer;

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
    private int pointsPerArea = 20;

    // show point cloud
    [SerializeField]
    private Toggle cbShowPointCloud;
    private bool showPointCloud = true;

    // 

    [SerializeField]
    private Transform hitLocation;

    [SerializeField]
    private PolySurface surface;
    private MeshCollider meshCollider;

    void Start()
    {
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

            if (Random.value < (pointsInTriangle - (int)pointsInTriangle)) {
                pointsInTriangle += 1;
            }
            // Debug.Log($"Tri {i / 3}: area = {area} ({area/totalSurfaceArea*100}% of total), generating {(int)pointsInTriangle} points");

            for (int k = 0; k < (int)pointsInTriangle; k++) {
                float r1 = Mathf.Sqrt(Random.Range(0.0f, 1.0f));
                float r2 = Random.Range(0.0f, 1.0f);
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

    private void OnRenderObject() {
        if (currentPointCloud == null)
            return;

        if (showPointCloud) {
            pointCloudMaterial.SetBuffer("pointCloud", pointCloudBuffer);
            pointCloudMaterial.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Points, 1, currentPointCloud.vertices.Length);
        } else {
            Graphics.DrawMesh(currentPointCloud.mesh, Matrix4x4.identity, meshMaterial, 0);
        }
    }

    private void SetActivePointCloud(PointCloud pc) {
        currentPointCloud = pc;
        pointCloudBuffer.SetData(currentPointCloud.vertices);
    }

    private void Update() {
        if (currentPointCloud == null)
            return;

        if (Input.GetMouseButtonDown(1)) {
            var ray = camera.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out var hit)) {
                camera.transform.parent.position = hit.point;
            } else {
                camera.transform.parent.position = Vector3.zero;
            }
        }
        if (Input.GetMouseButtonDown(0)) {
            var ray = camera.ScreenPointToRay(Input.mousePosition);
            meshCollider.sharedMesh = currentPointCloud.mesh;
            if (meshCollider.Raycast(ray, out var hit, float.MaxValue)) {
                hitLocation.position = hit.point;
                surface.RecalculateCoefficients(hit.point, currentPointCloud);
            }
        }
    }
}
