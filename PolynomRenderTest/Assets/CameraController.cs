using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class CameraController : MonoBehaviour
{
    [SerializeField]
    private float mouseSensitivity = 0.5f;
    [SerializeField]
    private float scrollSpeed = 0.5f;
    [SerializeField]
    private float zoomMax = 100;

    private Vector2 lastMousePos;

    [SerializeField]
    [Range(1, 100)]
    private float distance = 10;

    [SerializeField]
    [Range(-89, 89)]
    private float pitch = 0;
    [SerializeField]
    private float yaw = 0;

    [SerializeField]
    private Transform hitLocation;

    [SerializeField]
    private PolySurface surface;


    void Start()
    {
        lastMousePos = Input.mousePosition;
    }

    void Update()
    {
        var mouseScroll = Input.mouseScrollDelta;
        var mousePos = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
        var mouseMove = mousePos - lastMousePos;
        lastMousePos = mousePos;

        distance = Mathf.Clamp(distance / (1 + mouseScroll.y * scrollSpeed), 1, zoomMax);

        if (Input.GetMouseButtonDown(2))
            mouseMove *= 0;

        if (Input.GetMouseButton(2)) {
            yaw += mouseMove.x * mouseSensitivity;
            pitch = Mathf.Clamp(pitch - mouseMove.y * mouseSensitivity, -89, 89);
        }

        var rot = Quaternion.Euler(pitch, yaw, 0);
        transform.position = rot * Vector3.back * distance;
        transform.LookAt(Vector3.zero, Vector3.up);

        if (Input.GetMouseButtonDown(0)) {
            var ray = GetComponent<Camera>().ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out var hit)) {
                hitLocation.position = hit.point;

                var mesh = hit.collider.GetComponent<MeshCollider>().sharedMesh;
                surface.RecalculateCoefficients(hit.point, mesh);
            }
        }
    }
}
