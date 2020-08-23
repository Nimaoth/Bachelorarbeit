using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class AxisVisTest : MonoBehaviour
{
    [Range(0, 180)]
    public float alpha;

    [Range(0, 360)]
    public float beta;

    [Range(-1, 1)]
    public float g;
    [Range(0, 500)]
    public int randomVectorCount = 0;


    private void OnDrawGizmos()
    {
        Gizmos.DrawSphere(Vector3.zero, 0.5f);

        Vector3 d = transform.forward;

        Vector3 fx = d;
        Vector3 fy = TestDataGenerator.GetPerpendicular(d);
        Vector3 fz = Vector3.Cross(fx, fy);

        Gizmos.color = Color.red;
        Gizmos.DrawRay(Vector3.zero, fx * 10);
        Gizmos.color = Color.green;
        Gizmos.DrawRay(Vector3.zero, fy * 5);
        Gizmos.color = Color.blue;
        Gizmos.DrawRay(Vector3.zero, fz * 5);

        Gizmos.color = Color.white;

        var f = new Matrix4x4(fx.ToVector4(), fy.ToVector4(), fz.ToVector4(), new Vector4(0, 0, 0, 1));
        var df = new Vector3(Mathf.Cos(alpha * Mathf.Deg2Rad), Mathf.Sin(alpha * Mathf.Deg2Rad) * Mathf.Cos(beta * Mathf.Deg2Rad), Mathf.Sin(alpha * Mathf.Deg2Rad) * Mathf.Sin(beta * Mathf.Deg2Rad));
        var d_out = f * df;
        Gizmos.DrawRay(Vector3.zero, d_out * 3);

        var random = new System.Random();
        Gizmos.color = new Color(1, 1, 1, 0.5f);
        for (int i = 0; i < randomVectorCount; i++)
        {
            d_out = TestDataGenerator.GetRandomNewDirection(random, g, d);
            Gizmos.DrawRay(Vector3.zero, d_out * 3);
        }
    }
}

public static class Vector3Ext
{
    public static Vector4 ToVector4(this Vector3 self, float z = 0.0f)
    {
        return new Vector4(self.x, self.y, self.z, z);
    }
}