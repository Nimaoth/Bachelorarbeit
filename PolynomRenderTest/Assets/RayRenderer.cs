using MathNet.Numerics.LinearAlgebra.Complex;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

[ExecuteInEditMode]
public class RayRenderer : MonoBehaviour
{
    public Vector3 pos;
    public Vector3 dir;

    private LineRenderer lineRenderer;

    void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();
    }

    // Update is called once per frame
    void Update()
    {
        lineRenderer.SetPosition(0, pos);
        lineRenderer.SetPosition(1, pos + dir * 10);
    }
}
