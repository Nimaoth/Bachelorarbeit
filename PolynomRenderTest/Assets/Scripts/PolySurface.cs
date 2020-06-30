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
    public Transform rotationT;

    public static float MAX_DIST = 500;
    public static float MAX_STEPS = 500;
    public static float SURF_DIST = 0.0001f;
    public static float _StepSize = 0.01f;

    // surface/medium properties

    [SerializeField]
    [Range(-1.0f, 1.0f)]
    public float g = 0.9f;
    [SerializeField]
    [Min(0.0f)]
    public float ior = 1.0f;

    [SerializeField]
    [Min(0.0f)]
    public float sigmaS = 0.5f;
    [SerializeField]
    [Min(0.0f)]
    public float sigmaA = 0.5f;
    [SerializeField]
    [Min(0.0f)]
    public float sigmaT = 0.5f;

    [SerializeField]
    private float sigmaSred = 0.0f;
    public float SigmaSReduced => sigmaSred;
    [SerializeField]
    private float sigmaTred = 0.0f;
    [SerializeField]
    private float alphaRed = 0.0f;
    [SerializeField]
    private float alphaEff = 0.0f;
    [SerializeField]
    private float mad = 0.0f;
    [SerializeField]
    private float standardDeviation = 0.0f;
    public float StandardDeviation => standardDeviation;


    //

    [SerializeField]
    private new Camera camera;

    [SerializeField]
    private Material material;

    [SerializeField]
    private volatile float[] coefficients;
    public float[] Coefficients => coefficients;

    private Vector3 center;
    public Vector3 Center => center;

    private float[] coefficientsTemp = new float[20];

    private Vector3 rotRight;
    private Vector3 rotUp;
    private Vector3 rotForward;

    private void Start() {
        if (coefficients == null)
            coefficients = new float[20];
    }

    private void CalculateEffectiveProperties() {
        sigmaT = sigmaA + sigmaS;
        sigmaSred = (1 - g) * sigmaS;
        sigmaTred = sigmaA + sigmaSred;
        alphaRed = sigmaSred / sigmaTred;

        float e8 = Mathf.Pow((float)Math.E, 8);
        alphaEff = 1 - (0.125f * Mathf.Log(e8 + alphaRed * (1 - e8)));

        mad = MAD(g, alphaRed);

        standardDeviation = 2 * mad / sigmaTred;
    }

    private float MAD(float g, float a_red) {
        float e8 = Mathf.Pow((float)Math.E, 8);
        float a_eff = 1 - (0.125f * Mathf.Log(e8 + a_red * (1 - e8)));
        return 0.25f * g + 0.25f * a_red + a_eff;
    }

    public void SetCoefficients(float[] cos) {
        coefficients = cos;
        rotatePolynomial(coefficients, rotRight, rotUp, rotForward);
    }

    private void rotatePolynomial(float[] c, Vector3 s, Vector3 t, Vector3 n) {
        coefficientsTemp[0] = c[0];
        coefficientsTemp[1] = c[1]*s.x + c[2]*s.y + c[3]*s.z;
        coefficientsTemp[2] = c[1]*t.x + c[2]*t.y + c[3]*t.z;
        coefficientsTemp[3] = c[1]*n.x + c[2]*n.y + c[3]*n.z;
        coefficientsTemp[4] = c[4]*Mathf.Pow(s.x, 2) + c[5]*s.x*s.y + c[6]*s.x*s.z + c[7]*Mathf.Pow(s.y, 2) + c[8]*s.y*s.z + c[9]*Mathf.Pow(s.z, 2);
        coefficientsTemp[5] = 2*c[4]*s.x*t.x + c[5]*(s.x*t.y + s.y*t.x) + c[6]*(s.x*t.z + s.z*t.x) + 2*c[7]*s.y*t.y + c[8]*(s.y*t.z + s.z*t.y) + 2*c[9]*s.z*t.z;
        coefficientsTemp[6] = 2*c[4]*n.x*s.x + c[5]*(n.x*s.y + n.y*s.x) + c[6]*(n.x*s.z + n.z*s.x) + 2*c[7]*n.y*s.y + c[8]*(n.y*s.z + n.z*s.y) + 2*c[9]*n.z*s.z;
        coefficientsTemp[7] = c[4]*Mathf.Pow(t.x, 2) + c[5]*t.x*t.y + c[6]*t.x*t.z + c[7]*Mathf.Pow(t.y, 2) + c[8]*t.y*t.z + c[9]*Mathf.Pow(t.z, 2);
        coefficientsTemp[8] = 2*c[4]*n.x*t.x + c[5]*(n.x*t.y + n.y*t.x) + c[6]*(n.x*t.z + n.z*t.x) + 2*c[7]*n.y*t.y + c[8]*(n.y*t.z + n.z*t.y) + 2*c[9]*n.z*t.z;
        coefficientsTemp[9] = c[4]*Mathf.Pow(n.x, 2) + c[5]*n.x*n.y + c[6]*n.x*n.z + c[7]*Mathf.Pow(n.y, 2) + c[8]*n.y*n.z + c[9]*Mathf.Pow(n.z, 2);
        coefficientsTemp[10] = c[10]*Mathf.Pow(s.x, 3) + c[11]*Mathf.Pow(s.x, 2)*s.y + c[12]*Mathf.Pow(s.x, 2)*s.z + c[13]*s.x*Mathf.Pow(s.y, 2) + c[14]*s.x*s.y*s.z + c[15]*s.x*Mathf.Pow(s.z, 2) + c[16]*Mathf.Pow(s.y, 3) + c[17]*Mathf.Pow(s.y, 2)*s.z + c[18]*s.y*Mathf.Pow(s.z, 2) + c[19]*Mathf.Pow(s.z, 3);
        coefficientsTemp[11] = 3*c[10]*Mathf.Pow(s.x, 2)*t.x + c[11]*(Mathf.Pow(s.x, 2)*t.y + 2*s.x*s.y*t.x) + c[12]*(Mathf.Pow(s.x, 2)*t.z + 2*s.x*s.z*t.x) + c[13]*(2*s.x*s.y*t.y + Mathf.Pow(s.y, 2)*t.x) + c[14]*(s.x*s.y*t.z + s.x*s.z*t.y + s.y*s.z*t.x) + c[15]*(2*s.x*s.z*t.z + Mathf.Pow(s.z, 2)*t.x) + 3*c[16]*Mathf.Pow(s.y, 2)*t.y + c[17]*(Mathf.Pow(s.y, 2)*t.z + 2*s.y*s.z*t.y) + c[18]*(2*s.y*s.z*t.z + Mathf.Pow(s.z, 2)*t.y) + 3*c[19]*Mathf.Pow(s.z, 2)*t.z;
        coefficientsTemp[12] = 3*c[10]*n.x*Mathf.Pow(s.x, 2) + c[11]*(2*n.x*s.x*s.y + n.y*Mathf.Pow(s.x, 2)) + c[12]*(2*n.x*s.x*s.z + n.z*Mathf.Pow(s.x, 2)) + c[13]*(n.x*Mathf.Pow(s.y, 2) + 2*n.y*s.x*s.y) + c[14]*(n.x*s.y*s.z + n.y*s.x*s.z + n.z*s.x*s.y) + c[15]*(n.x*Mathf.Pow(s.z, 2) + 2*n.z*s.x*s.z) + 3*c[16]*n.y*Mathf.Pow(s.y, 2) + c[17]*(2*n.y*s.y*s.z + n.z*Mathf.Pow(s.y, 2)) + c[18]*(n.y*Mathf.Pow(s.z, 2) + 2*n.z*s.y*s.z) + 3*c[19]*n.z*Mathf.Pow(s.z, 2);
        coefficientsTemp[13] = 3*c[10]*s.x*Mathf.Pow(t.x, 2) + c[11]*(2*s.x*t.x*t.y + s.y*Mathf.Pow(t.x, 2)) + c[12]*(2*s.x*t.x*t.z + s.z*Mathf.Pow(t.x, 2)) + c[13]*(s.x*Mathf.Pow(t.y, 2) + 2*s.y*t.x*t.y) + c[14]*(s.x*t.y*t.z + s.y*t.x*t.z + s.z*t.x*t.y) + c[15]*(s.x*Mathf.Pow(t.z, 2) + 2*s.z*t.x*t.z) + 3*c[16]*s.y*Mathf.Pow(t.y, 2) + c[17]*(2*s.y*t.y*t.z + s.z*Mathf.Pow(t.y, 2)) + c[18]*(s.y*Mathf.Pow(t.z, 2) + 2*s.z*t.y*t.z) + 3*c[19]*s.z*Mathf.Pow(t.z, 2);
        coefficientsTemp[14] = 6*c[10]*n.x*s.x*t.x + c[11]*(2*n.x*s.x*t.y + 2*n.x*s.y*t.x + 2*n.y*s.x*t.x) + c[12]*(2*n.x*s.x*t.z + 2*n.x*s.z*t.x + 2*n.z*s.x*t.x) + c[13]*(2*n.x*s.y*t.y + 2*n.y*s.x*t.y + 2*n.y*s.y*t.x) + c[14]*(n.x*s.y*t.z + n.x*s.z*t.y + n.y*s.x*t.z + n.y*s.z*t.x + n.z*s.x*t.y + n.z*s.y*t.x) + c[15]*(2*n.x*s.z*t.z + 2*n.z*s.x*t.z + 2*n.z*s.z*t.x) + 6*c[16]*n.y*s.y*t.y + c[17]*(2*n.y*s.y*t.z + 2*n.y*s.z*t.y + 2*n.z*s.y*t.y) + c[18]*(2*n.y*s.z*t.z + 2*n.z*s.y*t.z + 2*n.z*s.z*t.y) + 6*c[19]*n.z*s.z*t.z;
        coefficientsTemp[15] = 3*c[10]*Mathf.Pow(n.x, 2)*s.x + c[11]*(Mathf.Pow(n.x, 2)*s.y + 2*n.x*n.y*s.x) + c[12]*(Mathf.Pow(n.x, 2)*s.z + 2*n.x*n.z*s.x) + c[13]*(2*n.x*n.y*s.y + Mathf.Pow(n.y, 2)*s.x) + c[14]*(n.x*n.y*s.z + n.x*n.z*s.y + n.y*n.z*s.x) + c[15]*(2*n.x*n.z*s.z + Mathf.Pow(n.z, 2)*s.x) + 3*c[16]*Mathf.Pow(n.y, 2)*s.y + c[17]*(Mathf.Pow(n.y, 2)*s.z + 2*n.y*n.z*s.y) + c[18]*(2*n.y*n.z*s.z + Mathf.Pow(n.z, 2)*s.y) + 3*c[19]*Mathf.Pow(n.z, 2)*s.z;
        coefficientsTemp[16] = c[10]*Mathf.Pow(t.x, 3) + c[11]*Mathf.Pow(t.x, 2)*t.y + c[12]*Mathf.Pow(t.x, 2)*t.z + c[13]*t.x*Mathf.Pow(t.y, 2) + c[14]*t.x*t.y*t.z + c[15]*t.x*Mathf.Pow(t.z, 2) + c[16]*Mathf.Pow(t.y, 3) + c[17]*Mathf.Pow(t.y, 2)*t.z + c[18]*t.y*Mathf.Pow(t.z, 2) + c[19]*Mathf.Pow(t.z, 3);
        coefficientsTemp[17] = 3*c[10]*n.x*Mathf.Pow(t.x, 2) + c[11]*(2*n.x*t.x*t.y + n.y*Mathf.Pow(t.x, 2)) + c[12]*(2*n.x*t.x*t.z + n.z*Mathf.Pow(t.x, 2)) + c[13]*(n.x*Mathf.Pow(t.y, 2) + 2*n.y*t.x*t.y) + c[14]*(n.x*t.y*t.z + n.y*t.x*t.z + n.z*t.x*t.y) + c[15]*(n.x*Mathf.Pow(t.z, 2) + 2*n.z*t.x*t.z) + 3*c[16]*n.y*Mathf.Pow(t.y, 2) + c[17]*(2*n.y*t.y*t.z + n.z*Mathf.Pow(t.y, 2)) + c[18]*(n.y*Mathf.Pow(t.z, 2) + 2*n.z*t.y*t.z) + 3*c[19]*n.z*Mathf.Pow(t.z, 2);
        coefficientsTemp[18] = 3*c[10]*Mathf.Pow(n.x, 2)*t.x + c[11]*(Mathf.Pow(n.x, 2)*t.y + 2*n.x*n.y*t.x) + c[12]*(Mathf.Pow(n.x, 2)*t.z + 2*n.x*n.z*t.x) + c[13]*(2*n.x*n.y*t.y + Mathf.Pow(n.y, 2)*t.x) + c[14]*(n.x*n.y*t.z + n.x*n.z*t.y + n.y*n.z*t.x) + c[15]*(2*n.x*n.z*t.z + Mathf.Pow(n.z, 2)*t.x) + 3*c[16]*Mathf.Pow(n.y, 2)*t.y + c[17]*(Mathf.Pow(n.y, 2)*t.z + 2*n.y*n.z*t.y) + c[18]*(2*n.y*n.z*t.z + Mathf.Pow(n.z, 2)*t.y) + 3*c[19]*Mathf.Pow(n.z, 2)*t.z;
        coefficientsTemp[19] = c[10]*Mathf.Pow(n.x, 3) + c[11]*Mathf.Pow(n.x, 2)*n.y + c[12]*Mathf.Pow(n.x, 2)*n.z + c[13]*n.x*Mathf.Pow(n.y, 2) + c[14]*n.x*n.y*n.z + c[15]*n.x*Mathf.Pow(n.z, 2) + c[16]*Mathf.Pow(n.y, 3) + c[17]*Mathf.Pow(n.y, 2)*n.z + c[18]*n.y*Mathf.Pow(n.z, 2) + c[19]*Mathf.Pow(n.z, 3);
        for (int i = 0; i < coefficientsTemp.Length; ++i) {
            c[i] = coefficientsTemp[i];
        }
    }

    public void SetCenter(Vector3 center) {
        this.center = center;
    }

    void Update()
    {
        rotRight = rotationT.right;
        rotUp = rotationT.up;
        rotForward = rotationT.forward;

        CalculateEffectiveProperties();

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

    public static Vector3 GetNormalAt(float[] coefficients, Vector3 pos, Vector3 center) {
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

    public static float EvaluateAt(float[] coefficients, Vector3 p, Vector3 center) {
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

    private static float BinarySearch(float[] coefficients, Vector3 ro, Vector3 rd, float t_min, float t_max, Vector3 center) {
        float d_min = EvaluateAt(coefficients, ro + rd * t_min, center);
        float d_max = EvaluateAt(coefficients, ro + rd * t_max, center);

        for(int i = 0; Mathf.Abs(d_min) >= SURF_DIST && i < MAX_STEPS / 8; i++) {
            float t = (t_min + t_max) * 0.5f;
            float d = EvaluateAt(coefficients, ro + rd * t, center);

            if (Mathf.Sign(d) == Mathf.Sign(d_min)) {
                t_min = t;
            } else {
                t_max = t;
            }
        }

        return t_max;
    }

    public static (float dist, bool inside) RayMarch(float[] coefficients, Vector3 ro, Vector3 rd, float t_max, Vector3 center) {
        float stepSize = _StepSize;

        float t_prev = 0;
        float t = stepSize;

        float prev_dist = EvaluateAt(coefficients, ro, center);
        float dist;

        bool inside = prev_dist < 0;

        for (int i = 0; true; i++) {
            if (i >= MAX_STEPS) {
                return (MAX_DIST + 1.0f, inside);
            }
            Vector3 p = ro + rd * t;
            dist = EvaluateAt(coefficients, p, center);

            if (t >= t_max) {
                return (t_max, EvaluateAt(coefficients, ro + rd * t_max, center) < 0);
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

        t = BinarySearch(coefficients, ro, rd, t_prev, t, center);

        return (t, !inside);
    }
}
