using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Single;
using System;

[ExecuteInEditMode]
public class PolySurface : MonoBehaviour
{
    [SerializeField]
    private new Camera camera;

    [SerializeField]
    private Material material;

    [SerializeField]
    private GameObject testPointPrefab;

    private List<GameObject> testPoints;
    private GameObject testPointParent;

    [SerializeField]
    private bool generateTestPoints;

    [SerializeField]
    private Color color;

    [SerializeField]
    private float boxSize;

    [SerializeField]
    private int maxVertexCount;
    [SerializeField]
    private float largeCoefficientPenalty = 0.0001f;
    [SerializeField]
    [Range(1.0f, 100.0f)]
    private float weightScale = 1.0f;

    [SerializeField]
    private float[] coefficients;

    private Vector3 currentCenter;

    private void Awake() {
        testPoints = new List<GameObject>();

        testPointParent = GameObject.Find("TEST POINTS");
        if (testPointParent == null)
            testPointParent = new GameObject("TEST POINTS");
    }

    void Update()
    {
        material.SetVector("cameraPos", camera.transform.position);
        material.SetVector("cameraDir", camera.transform.forward);
        material.SetVector("cameraRight", camera.transform.right);
        material.SetFloat("cameraFOV", camera.fieldOfView);
        material.SetFloat("boxSize", boxSize);
        material.SetColor("color", color);
        material.SetVector("center", currentCenter);
        material.SetFloatArray("coefficients", coefficients);

        var bl = camera.ViewportPointToRay(new Vector3(0, 0, 0));
        var br = camera.ViewportPointToRay(new Vector3(1, 0, 0));
        var tl = camera.ViewportPointToRay(new Vector3(0, 1, 0));
        var tr = camera.ViewportPointToRay(new Vector3(1, 1, 0));
        material.SetVector("cameraPosBL", bl.origin);
        material.SetVector("cameraPosBR", br.origin);
        material.SetVector("cameraPosTL", tl.origin);
        material.SetVector("cameraPosTR", tr.origin);
        material.SetVector("cameraDirBL", bl.direction);
        material.SetVector("cameraDirBR", br.direction);
        material.SetVector("cameraDirTL", tl.direction);
        material.SetVector("cameraDirTR", tr.direction);
    }

    public void RecalculateCoefficients(Vector3 center, PointCloud pointCloud)
    {
        foreach (var t in testPoints)
            GameObject.Destroy(t);
        testPoints.Clear();

        currentCenter = center;

        Vertex[] vertices = pointCloud.vertices;
        vertices = vertices
            .OrderBy(v => (v.position - center).sqrMagnitude)
            .Take(Mathf.Min(vertices.Length, maxVertexCount))
            .Select(v => new Vertex(v.position - center, v.normal))
            .ToArray();

        if (generateTestPoints) {
            foreach (var v in vertices) {
                if (UnityEngine.Random.Range(0.0f, 1.0f) < 500.0f / (float)maxVertexCount) {
                    var go = GameObject.Instantiate(testPointPrefab, v.position + center, Quaternion.identity, testPointParent.transform);
                    go.transform.localScale = weight(v.position) * go.transform.localScale;
                    testPoints.Add(go);
                }
            }
        }

        int m = vertices.Length;

        var X = DenseMatrix.OfRows(m, 20, vertices.Select(v => Pc(v.position)));
        var tmp = 2 * X.Transpose() * X + Sum(20, 20, vertices, v => weight(v.position) * Y(v.position)) + 2 * largeCoefficientPenalty;
        var c = tmp.Inverse() * Sum(20, 1, vertices, v => weight(v.position) * Z(v.position, v.normal));

        coefficients = c.Column(0).ToArray();
        coefficients[0] = 0;
    }

    private DenseMatrix Sum(int r, int c, Vertex[] vertices, Func<Vertex, DenseMatrix> func) {
        DenseMatrix result = new DenseMatrix(r, c);
        for (int i = 0; i < vertices.Length; i++) {
            result += func(vertices[i]);
        }
        return result;
    }

    private float weight(Vector3 p) {
        return Mathf.Exp(-0.5f * p.sqrMagnitude * weightScale);
    }

    private float[] Pc(Vector3 p) {
        float w = Mathf.Sqrt(weight(p));
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

    private DenseMatrix Y(Vector3 p)
    {
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
