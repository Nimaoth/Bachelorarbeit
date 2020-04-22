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
    public float MAX_DIST = 500;
    public float MAX_STEPS = 500;
    public float STEP_SIZE = 0.1f;
    public float SURF_DIST = 0.0001f;
    public float _StepSize = 0.1f;

    [SerializeField]
    public float sigmaS = 0.5f;
    [SerializeField]
    public float sigmaA = 0.5f;
    [SerializeField]
    public float sigmaT = 0.5f;
    [SerializeField]
    [Range(-1.0f, 1.0f)]
    public float g = 0.9f;

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

    public Vector3 GetNormalAt(Vector3 pos) {
        // specifically for poly surface
        pos -= center;
        float x = pos.x;
        float y = pos.y;
        float z = pos.z;
        float a = coefficients[0];
        float b = coefficients[1];
        float c = coefficients[2];
        float d = coefficients[3];
        float e = coefficients[4];
        float f = coefficients[5];
        float g = coefficients[6];
        float h = coefficients[7];
        float i = coefficients[8];
        float j = coefficients[9];
        float k = coefficients[10];
        float l = coefficients[11];
        float m = coefficients[12];
        float n = coefficients[13];
        float o = coefficients[14];
        float p = coefficients[15];
        float q = coefficients[16];
        float r = coefficients[17];
        float s = coefficients[18];
        float t = coefficients[19];
        return new Vector3(
            2*e*x + 2*l*x*y + 2*m*x*z + 3*k*x*x + b + f*y + g*z + n*y*y + o*y*z + p*z*z,
            2*h*y + 2*n*x*y + 2*r*y*z + 3*q*y*y + c + f*x + i*z + l*x*x + o*x*z + s*z*z,
            2*j*z + 2*p*x*z + 2*s*y*z + 3*t*z*z + d + g*x + i*y + m*x*x + o*x*y + r*y*y
        ).normalized;
    }

    public float EvaluateAt(Vector3 p) {
        p -= center;
        float x = p.x;
        float y = p.y;
        float z = p.z;
        return
            coefficients[0] +
            coefficients[1] * x +
            coefficients[2] * y +
            coefficients[3] * z +
            coefficients[4] * x * x +
            coefficients[5] * x * y +
            coefficients[6] * x * z +
            coefficients[7] * y * y +
            coefficients[8] * y * z +
            coefficients[9] * z * z +
            coefficients[10] * x * x * x +
            coefficients[11] * x * x * y +
            coefficients[12] * x * x * z +
            coefficients[13] * x * y * y +
            coefficients[14] * x * y * z +
            coefficients[15] * x * z * z +
            coefficients[16] * y * y * y +
            coefficients[17] * y * y * z +
            coefficients[18] * y * z * z +
            coefficients[19] * z * z * z;
    }

    private float BinarySearch(Vector3 ro, Vector3 rd, float t_min, float t_max) {
        float d_min = EvaluateAt(ro + rd * t_min);
        float d_max = EvaluateAt(ro + rd * t_max);

        for(int i = 0; Mathf.Abs(d_min) >= SURF_DIST && i < MAX_STEPS / 8; i++) {
            float t = (t_min + t_max) * 0.5f;
            float d = EvaluateAt(ro + rd * t);

            if (Mathf.Sign(d) == Mathf.Sign(d_min)) {
                t_min = t;
            } else {
                t_max = t;
            }
        }

        return t_max;
    }

    public (float dist, bool inside) RayMarch(Vector3 ro, Vector3 rd, float t_max) {
        float stepSize = _StepSize;

        float t_prev = 0;
        float t = stepSize;

        float prev_dist = EvaluateAt(ro);
        float dist;

        bool inside = prev_dist < 0;

        for (int i = 0; true; i++) {
            if (i >= MAX_STEPS) {
                return (MAX_DIST + 1.0f, inside);
            }
            Vector3 p = ro + rd * t;
            dist = EvaluateAt(p);

            if (t >= t_max) {
                return (t_max, EvaluateAt(ro + rd * t_max) < 0);
            }

            if (inside && dist >= 0) {
                break;
            } else if (!inside && dist < 0) {
                break;
            }

            prev_dist = dist;
            t_prev = t;
            t += stepSize;
            stepSize = Mathf.Lerp(_StepSize, _StepSize * 10, t / MAX_DIST);

            if (t > MAX_DIST)
                return (t, inside);
        }

        t = BinarySearch(ro, rd, t_prev, t);

        return (t, !inside);
    }
}
