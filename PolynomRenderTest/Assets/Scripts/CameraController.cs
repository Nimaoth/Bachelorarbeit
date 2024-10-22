﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class CameraController : MonoBehaviour
{
    [SerializeField]
    private new Camera camera;
    
    [SerializeField]
    private float mouseSensitivity = 0.5f;
    [SerializeField]
    private float scrollSpeed = 0.5f;
    [SerializeField]
    private float zoomMax = 100;

    private Vector2 lastMousePos;

    [SerializeField]
    [Range(0.01f, 100)]
    private float distance = 10;

    [SerializeField]
    [Range(-89, 89)]
    private float pitch = 0;
    [SerializeField]
    private float yaw = 0;

    [SerializeField]
    private float panSpeed = 1;

    void Start()
    {
        lastMousePos = Input.mousePosition;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F1))
        {
            PlayerPrefs.SetFloat("distance", distance);
            PlayerPrefs.SetFloat("pitch", pitch);
            PlayerPrefs.SetFloat("yaw", yaw);
            PlayerPrefs.SetFloat("cam_x", transform.position.x);
            PlayerPrefs.SetFloat("cam_y", transform.position.y);
            PlayerPrefs.SetFloat("cam_z", transform.position.z);
            PlayerPrefs.Save();
        }
        if (Input.GetKeyDown(KeyCode.F2))
        {
            distance = PlayerPrefs.GetFloat("distance");
            pitch = PlayerPrefs.GetFloat("pitch");
            yaw = PlayerPrefs.GetFloat("yaw");
            transform.position = new Vector3(PlayerPrefs.GetFloat("cam_x"), PlayerPrefs.GetFloat("cam_y"), PlayerPrefs.GetFloat("cam_z"));
        }

        var shift = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        var mouseScroll = Input.mouseScrollDelta;
        var mousePos = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
        var mouseMove = mousePos - lastMousePos;
        lastMousePos = mousePos;

        distance = Mathf.Clamp(distance / (1 + mouseScroll.y * scrollSpeed), 0.01f, zoomMax);

        if (Input.GetMouseButtonDown(1) || Input.GetMouseButtonDown(2))
            mouseMove *= 0;

        if (shift && Input.GetMouseButton(2)) {
            Vector3 move = Vector3.zero;
            move -= camera.transform.up * mouseMove.y * panSpeed * distance * 0.005f;
            move -= camera.transform.right * mouseMove.x * panSpeed * distance * 0.005f;
            transform.position += move;
        }

        if (!shift && Input.GetMouseButton(2)) {
            yaw += mouseMove.x * mouseSensitivity;
            pitch = Mathf.Clamp(pitch - mouseMove.y * mouseSensitivity, -89, 89);
        }

        var rot = Quaternion.Euler(pitch, yaw, 0);
        camera.transform.localPosition = rot * Vector3.back * distance;
        camera.transform.LookAt(transform.position, Vector3.up);
    }
}
