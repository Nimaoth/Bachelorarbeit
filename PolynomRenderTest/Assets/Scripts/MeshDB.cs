using System;
using System.CodeDom;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Complex.Solvers;
using MathNet.Numerics.LinearAlgebra.Single;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UI;

public enum SampleSource
{
    Generate,
    Python,
    Cpp
}

public struct MeshDBSample
{
    public Vector3 position;
    public Vector3 gradient;
}

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
    public MyMesh mesh;
    public Vertex[] vertices;

    public PointCloud(MyMesh mesh) {
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

public class MyMesh
{
    public Vector3[] positions;
    public Vector3[] normals;
    public float[][] coefficients;
    public int[] faces;

    public Mesh mesh;

    public MyMesh(Mesh mesh)
    {
        this.mesh = mesh;
        positions = mesh.vertices;
        normals = mesh.normals;
        faces = mesh.GetIndices(0);
        coefficients = null;
    }

    public MyMesh(Vector3[] positions, Vector3[] normals, float[][] coefficients, int[] faces)
    {
        mesh = new Mesh();

        mesh.vertices = this.positions = positions;
        mesh.normals = this.normals = normals;
        this.faces = faces;
        mesh.SetIndices(faces, MeshTopology.Triangles, 0);
        this.coefficients = coefficients;
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
    [Range(1.0f, 200.0f)]
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
    private Vector3 currentCenter, currentNormal, currentDirection;
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
    public int samplesPerPoint = 100;

    [Range(0, 1)]
    public float trainingProgress = 0.0f;

    MyMesh LoadObj(string path)
    {
        var positions = new List<Vector3>();
        var normals = new List<Vector3>();
        var coefficients = new List<float[]>();
        var faces = new List<int>();

        foreach (string line in File.ReadLines(path))
        {
            if (line.StartsWith("v "))
            {
                var nums = line.Substring(2).Split(' ').Select(str => float.Parse(str)).ToArray();
                positions.Add(new Vector3(nums[0], nums[1], nums[2]));
            }
            else if (line.StartsWith("vn "))
            {
                var nums = line.Substring(3).Split(' ').Select(str => float.Parse(str)).ToArray();
                normals.Add(new Vector3(nums[0], nums[1], nums[2]));
            }
            else if (line.StartsWith("vc "))
            {
                var nums = line.Substring(3).Split(' ').Select(str => float.Parse(str)).ToArray();
                coefficients.Add(nums);
            }
            else if (line.StartsWith("f "))
            {
                var indices = line.Substring(2).Split(' ').Select(str => int.Parse(str.Split('/')[0]) - 1).ToArray();
                faces.AddRange(indices);
            }
        }

        return new MyMesh(positions.ToArray(), normals.ToArray(), coefficients.ToArray(), faces.ToArray());
    }

    public string objFilePath = @"D:\New folder\bunny_c.obj";

    public string samplesPath = "";
    public string modelName = "";
    public string modelVersion = "final";
    public float modelStddev = 20.0f;
    public bool incidentIsNormal = true;

    public GameObject samplePointPrefab;
    private List<GameObject> samples = new List<GameObject>();

    public bool autoProjectSamples = false;
    public bool renderPolySurface = true;
    public SampleSource sampleSource = SampleSource.Generate;

    // methods

    void Start()
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        meshCollider = GetComponent<MeshCollider>();
        pointClouds = new PointCloud[objects.Length + 1];

        pointClouds[0] = new PointCloud(LoadObj(objFilePath));
        var button0 = GameObject.Instantiate(meshSelectorButtonPrefab, meshSelectorButtonList.transform);
        button0.GetComponentInChildren<Text>().text = Path.GetFileName(objFilePath);
        button0.GetComponent<Button>().onClick.AddListener(() => {
            if (pointClouds[0].vertices == null)
            {
                ComputePoints(pointClouds[0]);
            }
            SetActivePointCloud(pointClouds[0]);
        });

        for (int i = 0; i < objects.Length; i++) {
            var mesh = objects[i];
            var pointCloud = pointClouds[i + 1] = new PointCloud(new MyMesh(mesh));

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
        showPointCloud = cbShowPointCloud.isOn;

        cbShowMesh.onValueChanged.AddListener(val => {
            showMesh = val;
        });
        showMesh = cbShowMesh.isOn;

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

        minimumPointCount = int.Parse(ifMinimumPointCount.text);
        pointsPerAreaMultiplier = int.Parse(ifPointsPerArea.text);

        ComputePoints(pointClouds[0]);
        SetActivePointCloud(pointClouds[0]);

        thread = new Thread(RecalculateCoefficientsThread);
        thread.Start();
    }

    void OnApplicationQuit() {
        try
        {
            pointCloudBuffer?.Release();
        }
        catch (Exception e)
        {
            Debug.LogError(e.ToString());
        }
    }

    private void OnRenderObject() {
        if (currentPointCloud == null)
            return;

        if (showMesh) {
            meshMaterial.SetPass(0);
            Graphics.DrawMeshNow(currentPointCloud.mesh.mesh, Matrix4x4.identity);
            // Graphics.DrawMesh(currentPointCloud.mesh, Matrix4x4.identity, meshMaterial, 0);
        }

        if (showPointCloud) {
            pointCloudMaterial.SetBuffer("pointCloud", pointCloudBuffer);
            pointCloudMaterial.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Points, 1, currentPointCloud.vertices.Length);
        }
    }

    System.Random samplesRandom = new System.Random();
    List<MeshDBSample> samplesTemp = new List<MeshDBSample>();
    System.Diagnostics.Process process = null;

    public void GetSamplesFromModelCurrentPos()
    {
        var dir = currentDirection;
        if (incidentIsNormal)
            dir = currentNormal;
        GetSamplesFromModel(currentCenter, currentNormal, dir);
    }

    private void GetSamplesFromModel(Vector3 center, Vector3 normal, Vector3 direction)
    {
        Debug.Log("Start generator");
        foreach (var go in samples)
            GameObject.Destroy(go);
        samples.Clear();

        var coefficients = surface.Coefficients.ToArray();

        var normalToZ = new Frame(normal);
        var zToDirection = new Frame(-direction).Invert();
        TestDataGenerator.RotatePolynomial(coefficients, normalToZ.x, normalToZ.y, normalToZ.z);

        if (!incidentIsNormal)
            TestDataGenerator.RotatePolynomial(coefficients, zToDirection.x, zToDirection.y, zToDirection.z);

        string[] arguments = null;
        string fileExe = null;

        var model = string.IsNullOrWhiteSpace(modelVersion) ? $"{modelName}/final" : $"{modelName}/{modelVersion}";

        switch (sampleSource)
        {
            case SampleSource.Cpp:
                fileExe = @"D:\Bachelorarbeit\CppModelTest\x64\Release\CppModelTest.exe";
                arguments = new string[]
                {
                            model,
                            samplesToGenerate.ToString(),
                            modelStddev.ToString(),
                            $"{string.Join(",", coefficients)},{surface.alphaEff},{surface.g},{surface.ior}"
                };
                break;
            case SampleSource.Python:
                fileExe = "python.exe";
                arguments = new string[]
                {
                            @"D:\Bachelorarbeit\Bachelorarbeit\Model\generate_samples.py",
                            model,
                            samplesToGenerate.ToString(),
                            modelStddev.ToString(),
                            $"{string.Join(",", coefficients)},{surface.alphaEff},{surface.g},{surface.ior}"
                };
                break;

            default:
                fileExe = @"D:\Bachelorarbeit\CppModelTest\x64\Release\CppModelTest.exe";
                arguments = new string[]
                {
                            model,
                            samplesToGenerate.ToString(),
                            modelStddev.ToString(),
                            $"{string.Join(",", coefficients)},{surface.alphaEff},{surface.g},{surface.ior}"
                };
                break;
        }

        var p = new System.Diagnostics.Process();
        p.StartInfo.FileName = fileExe;
        p.StartInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
        p.StartInfo.WorkingDirectory = @"D:\Bachelorarbeit\Bachelorarbeit\Model";
        p.StartInfo.UseShellExecute = false;
        p.StartInfo.CreateNoWindow = true;
        p.StartInfo.RedirectStandardOutput = true;
        p.StartInfo.Arguments = string.Join(" ", arguments.Select(a => $"\"{a}\""));

        Debug.Log(p.StartInfo.Arguments);

        var r = new System.Random();
        p.OutputDataReceived += (sender, args) => {
            if (args.Data != null && args.Data.StartsWith("# "))
            {
                var parts = args.Data.Substring(2).Split(',').Select(s => float.Parse(s)).ToArray();
                var pos = new Vector3(parts[0], parts[1], parts[2]);
                var n = PolySurface.GetNormalAt(coefficients, pos, Vector3.zero);

                //pos = new Vector3(0, 0, (float)r.NextDouble());

                if (!incidentIsNormal)
                {
                    pos = zToDirection.ToMatrix() * pos;
                    n = zToDirection.ToMatrix() * n;
                }
                pos = normalToZ.ToMatrix() * pos;
                n = normalToZ.ToMatrix() * n;

                pos += center;

                samplesTemp.Add(new MeshDBSample
                {
                    position = pos,
                    gradient = n
                });
            }
        };

        p.Start();
        p.BeginOutputReadLine();

        process = p;
    }

    public int samplesToGenerate = 1000;


    private int[] modelVersions = new int[] { 1, 2, 5, 10, 15, 20, 25 };
    private int currentModelVersionIndex = 0;

    private string[] modelNames = new string[] {
        "",
        "test-2020-08-04--19-09-47",
        "test-2020-08-04--21-06-49",
        "test-2020-08-05--15-44-24",
        "test-2020-08-04--15-05-38",
        "test-2020-08-04--16-45-37",
        "test-2020-08-04--22-49-23",
    };
    private int currentModelIndex = 0;

    public static void SaveVector(string name, Vector3 vec)
    {
        PlayerPrefs.SetFloat($"{name}_x", vec.x);
        PlayerPrefs.SetFloat($"{name}_y", vec.y);
        PlayerPrefs.SetFloat($"{name}_z", vec.z);
    }
    public static Vector3 LoadVector(string name)
    {
        return new Vector3(PlayerPrefs.GetFloat($"{name}_x"), PlayerPrefs.GetFloat($"{name}_y"), PlayerPrefs.GetFloat($"{name}_z"));
    }

    private void Update()
    {

        if (Input.GetKeyDown(KeyCode.F1))
        {
            SaveVector("currentCenter", currentCenter);
            SaveVector("currentNormal", currentNormal);
            PlayerPrefs.Save();
        }
        if (Input.GetKeyDown(KeyCode.F2))
        {
            currentCenter = LoadVector("currentCenter");
            currentNormal = LoadVector("currentNormal");

            hitLocation.transform.position = currentCenter;
        }

        if (Input.GetKeyDown(KeyCode.F4))
        {
            if (modelStddev == 1)
                modelStddev = 20;
            else
                modelStddev = 1;
        }

        if (Input.GetKeyDown(KeyCode.F5))
        {
            foreach (var go in samples)
                GameObject.Destroy(go);
            samples.Clear();
            GenerateSamples(samplesToGenerate);
        }
        if (Input.GetKeyDown(KeyCode.F6))
        {
            modelName = modelNames[currentModelIndex];
            currentModelIndex = (currentModelIndex+ 1) % modelNames.Length;
        }

        if (Input.GetKeyDown(KeyCode.F7))
        {
            modelVersion = modelVersions[currentModelVersionIndex].ToString();
            currentModelVersionIndex = (currentModelVersionIndex + 1) % modelVersions.Length;
        }
        if (Input.GetKeyDown(KeyCode.F8))
        {
            foreach (var go in samples)
                GameObject.Destroy(go);
            samples.Clear();
            GetSamplesFromModelCurrentPos();
        }

        surface.renderer.enabled = renderPolySurface;

        var ctrl = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);
        var shift = Input.GetKey(KeyCode.LeftShift);
            
        pointCloudMaterial.SetFloat("_WeightScale", weightScale);
        pointCloudMaterial.SetFloat("_WeightFactor", weightFactor);

        if (currentPointCloud != null)
        {
            currentVertices = currentPointCloud.vertices;

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
                meshCollider.sharedMesh = currentPointCloud.mesh.mesh;
                if (meshCollider.Raycast(ray, out var hit, float.MaxValue)) {
                    //Debug.Log($"({ray.origin.x}, {ray.origin.y}, {ray.origin.z}), ({ray.direction.x}, {ray.direction.y}, {ray.direction.z})");
                    hitLocation.position = hit.point;
                    currentCenter = hit.point;
                    currentNormal = hit.normal;
                    currentVertices = currentPointCloud.vertices;
                    pointCloudMaterial.SetVector("center", hit.point);

                    //
                    if (currentPointCloud.mesh.coefficients != null)
                    {
                        var uvw = hit.barycentricCoordinate;

                        var coefficientsArray = currentPointCloud.mesh.coefficients;
                        var faces = currentPointCloud.mesh.faces;

                        float[] c0 = coefficientsArray[faces[hit.triangleIndex * 3 + 0]];
                        float[] c1 = coefficientsArray[faces[hit.triangleIndex * 3 + 1]];
                        float[] c2 = coefficientsArray[faces[hit.triangleIndex * 3 + 2]];

                        float[] coefficients = new float[20];
                        for (int i = 0; i < 20; i++)
                        {
                            coefficients[i] = uvw.x * c0[i] + uvw.y * c1[i] + uvw.z * c2[i];
                        }

                        surface.SetCenter(currentCenter);
                        surface.SetCoefficients(coefficients);
                    }
                }
            }
            
        }

        if (ctrl && Input.GetMouseButtonDown(0)) {
            var point = camera.ScreenPointToRay(Input.mousePosition);

            var random = new System.Random();

            currentPath.Clear();
            currentPath.Add(new PathPoint
            {
                position = point.origin,
                direction = point.direction
            });


            var (dist, _) = PolySurface.RayMarch(coefficients, point.origin, point.direction, 1000.0f, surface.Center);
            var hit = point.origin + (dist + 0.001f) * point.direction;

            int stepCount = 0;
            var exit = TestDataGenerator.SimulatePath(
                random,
                surface.Coefficients,
                hit, point.direction,
                surface.Center,
                surface.g, surface.SigmaSReduced, surface.sigmaT,
                distScale,
                (p) => currentPath.Add(p),
                ref stepCount);

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

        if (process != null && process.HasExited)
        {
            Debug.Log("generator done");

            foreach (var go in samples)
                GameObject.Destroy(go);
            samples.Clear();

            foreach (var sample in samplesTemp)
            {
                var go = GameObject.Instantiate(samplePointPrefab, transform);
                go.transform.position = sample.position;
                samples.Add(go);
            }
            samplesTemp.Clear();
            process = null;
        }

        // generate samples
        if (shift && Input.GetMouseButtonDown(0))
        {
            foreach (var go in samples)
                GameObject.Destroy(go);
            samples.Clear();
            if (sampleSource != SampleSource.Generate && process == null)
            {
                var ray = camera.ScreenPointToRay(Input.mousePosition);
                if (meshCollider.Raycast(ray, out var hit, float.MaxValue))
                {
                    var dir = ray.direction;
                    if (incidentIsNormal)
                        dir = -hit.normal;
                    GetSamplesFromModel(hit.point, hit.normal, dir);
                }
            }
        }
        if (shift && Input.GetMouseButton(0))
        {
            if (sampleSource == SampleSource.Generate)
                GenerateSamples();
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
    
    public void LoadSamplesFromFile()
    {
        //currentVertices = null;

        //using (var file = File.OpenRead($"../train_data/{samplesPath}.json"))
        //{
        //    JsonSerializer serializer = new JsonSerializer();
        //    var reader = new JsonTextReader(new StreamReader(new BufferedStream(file)));
        //    List<Sample> samples = serializer.Deserialize<List<Sample>>(reader);

        //    var s0 = samples[0];
        //    var normalToZ = new Frame(s0.normal.ToVector3()).Invert();
        //    var coefficients = s0.features.Take(20).ToArray();
        //    TestDataGenerator.RotatePolynomial(coefficients, normalToZ.x, normalToZ.y, normalToZ.z);

        //    currentCenter = s0.point.ToVector3();
        //    currentNormal = s0.normal.ToVector3();
        //    surface.SetCoefficients(coefficients);
        //    surface.SetCenter(currentCenter);
        //    hitLocation.position = currentCenter;


                //foreach (var go in samples)
                //    GameObject.Destroy(go);
        //    this.samples.Clear();
        //    foreach (var s in samples.Take(Math.Min(100, samples.Count)))
        //    {
        //        var pos = new Vector3(s.out_pos[0], s.out_pos[1], s.out_pos[2]);
        //        pos = normalToZ.Invert().ToMatrix() * pos;
        //        pos += s.point.ToVector3();

        //        this.samples.Add(new MeshDBSample
        //        {
        //            position = pos,
        //            gradient = currentNormal,
        //        });
        //    }
        //}
    }

    private void GenerateSamples(int amount = 10)
    {
        meshCollider.sharedMesh = currentPointCloud.mesh.mesh;

        var camRay = camera.ScreenPointToRay(Input.mousePosition);
        var dir = camRay.direction;

        var (dist, _) = PolySurface.RayMarch(coefficients, camRay.origin, dir, 1000.0f, surface.Center);
        var hit = camRay.origin + (dist + 0.001f) * camRay.direction;

        hit = currentCenter - currentNormal * 0.001f;
        dir = -currentNormal;

        int count = 0;
        while (count < amount)
        {
            int stepCount = 0;
            var exit = TestDataGenerator.SimulatePath(
                samplesRandom,
                surface.Coefficients,
                hit, dir,
                surface.Center,
                surface.g, surface.SigmaSReduced, surface.sigmaT,
                distScale,
                null,
                ref stepCount);

            if (exit != null)
            {
                count += 1;
                var ray = new Ray(exit.Value, -PolySurface.GetNormalAt(surface.Coefficients, exit.Value, surface.Center));
                if (autoProjectSamples && meshCollider.Raycast(ray, out var h, float.MaxValue))
                {
                    var go = GameObject.Instantiate(samplePointPrefab, transform);
                    go.transform.position = h.point;
                    samples.Add(go);

                    //samples.Add(new MeshDBSample
                    //{
                    //    position = h.point,
                    //    gradient = -ray.direction,
                    //});
                }
                else
                {
                    var go = GameObject.Instantiate(samplePointPrefab, transform);
                    go.transform.position = exit.Value;
                    samples.Add(go);
                    //samples.Add(new MeshDBSample
                    //{
                    //    position = exit.Value,
                    //    gradient = -ray.direction
                    //});
                }
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

        var mesh = currentPointCloud.mesh.mesh;
        var vertices = mesh.vertices;
        var normals = mesh.normals;
        var indices = mesh.triangles;

        var builder = new StringBuilder();
        builder.AppendLine("o " + mesh.name);

        foreach (var v in vertices)
            builder.AppendLine($"v {v.x} {v.y} {v.z}");
        foreach (var v in normals)
            builder.AppendLine($"vn {v.x} {v.y} {v.z}");

        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        foreach (var coefficients in vertices.AsParallel().AsOrdered().Select(v => TestDataGenerator.CalculateCoefficients(
            new System.Random(42069),
            currentPointCloud.vertices,
            v,
            1,
            weightFactor,
            weightScale,
            maxVertexCount,
            randomVertexCount,
            randomVertexWeight,
            largeCoefficientPenalty)).AsSequential())
        {
            builder.Append($"vc");
            foreach (float c in coefficients)
                builder.Append(" " + c);
            builder.AppendLine();
        }
        stopwatch.Stop();
        Debug.Log("Parallel loop: " + stopwatch.Elapsed);

        //stopwatch.Restart();
        //var random = new System.Random();
        //foreach (var v in vertices)
        //{
        //    // calculate coefficients for random point on surfice
        //    float[] coefficients = TestDataGenerator.CalculateCoefficients(
        //        random,
        //        currentPointCloud.vertices,
        //        v,
        //        1,
        //        weightFactor,
        //        weightScale,
        //        maxVertexCount,
        //        randomVertexCount,
        //        randomVertexWeight,
        //        largeCoefficientPenalty);

        //    builder.Append($"vc");
        //    foreach (float c in coefficients)
        //        builder.Append(" " + c);
        //    builder.AppendLine();
        //}
        //stopwatch.Stop();
        //Debug.Log("Serial loop: " + stopwatch.Elapsed);

        string format = "f {0}//{0}/{0} {1}//{1}/{1} {2}//{2}/{2}\n";

        for (int i = 0; i < indices.Length; i += 3)
        {
            int i0 = indices[i] + 1;
            int i1 = indices[i + 1] + 1;
            int i2 = indices[i + 2] + 1;

            builder.AppendFormat(format, i0, i1, i2);
        }

        File.WriteAllText(mesh.name + ".obj", builder.ToString());
        Debug.Log("Saving mesh done");
    }

    public void SimulatePathRandomPath() {
        TestDataGenerator.GenerateTestData(
            currentPointCloud.mesh.mesh,
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
            samplesPerPoint,
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
        pointCloud.vertices = TestDataGenerator.ComputePointCloud(random, pointCloud.mesh.mesh.vertices, pointCloud.mesh.normals, pointCloud.mesh.mesh.triangles, minimumPointCount, pointsPerAreaMultiplier, surface.StandardDeviation);
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

        while (true) {
            watch.Restart();
            if (currentVertices?.Length > 0 && currentPointCloud.mesh.coefficients == null)
            {
                var random = new System.Random(randomVertexSeed);
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


    public void ProjectSamplesOnSurface()
    {
        //if (samples != null)
        //{
        //    meshCollider.sharedMesh = currentPointCloud.mesh.mesh;
        //    samples = samples.Select(s =>
        //    {
        //        var ray = new Ray(s.transform.position, -s.gradient);
        //        if (meshCollider.Raycast(ray, out var hit, float.MaxValue))
        //        {
        //            return new MeshDBSample
        //            {
        //                position = hit.point,
        //                gradient = s.gradient,
        //            };
        //        }
        //        else
        //        {
        //            return s;
        //        }
        //    }).ToList();
        //}
    }

    private void OnDrawGizmos()
    {
        if (samples != null)
        {
            var c = Gizmos.color;
            Gizmos.color = new Color(1, 0, 1, 0.5f);
            foreach (var sample in samples)
            {
                Gizmos.DrawSphere(sample.transform.position, 0.001f);
                //Gizmos.DrawRay(sample.position, sample.gradient * 0.005f);
            }
            Gizmos.color = c;
        }
    }
}
