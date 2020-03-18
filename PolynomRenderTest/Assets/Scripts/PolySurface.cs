using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra.Single;
using System;
using System.Threading;
using System.Diagnostics;

public class PolySurface : MonoBehaviour
{
    [SerializeField]
    private new Camera camera;

    [SerializeField]
    private Material material;

    [SerializeField]
    private volatile float[] coefficients;

    private Vector3 center;

    private void Start() {
        if (coefficients == null)
            coefficients = new float[20];
    }

    public void SetCoefficients(float[] cos) {
        coefficients = cos;
    }

    public void SetCenter(Vector3 center) {
        this.center = center;
    }

    void Update()
    {
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
        
        material.SetVector("cameraPos", camera.transform.position);
        material.SetVector("cameraDir", camera.transform.forward);
        material.SetVector("cameraRight", camera.transform.right);
        material.SetFloat("cameraFOV", camera.fieldOfView);
        material.SetVector("center", center);
        material.SetFloatArray("coefficients", coefficients);
    }
}
