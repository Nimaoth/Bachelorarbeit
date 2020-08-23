using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[UnityEditor.CustomEditor(typeof(MeshDB))]
public class MeshDBInspector : UnityEditor.Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        serializedObject.Update();

        var meshdb = target as MeshDB;

        if (GUILayout.Button("Load samples from file"))
            meshdb.LoadSamplesFromFile();
        if (GUILayout.Button("Project samples on surface"))
            meshdb.ProjectSamplesOnSurface();
        if (GUILayout.Button("Get samples from model"))
            meshdb.GetSamplesFromModelCurrentPos();

        if (GUILayout.Button("Generate Samples"))
            meshdb.SimulatePathRandomPath();

        if (GUILayout.Button("Generate Mesh"))
            meshdb.GenerateMeshWithCoefficients();



        if (GUILayout.Button("Save screenshot"))
        {
            Camera.main.GetComponent<Screenshot>().TakeScreenshot();
        }
    }
}