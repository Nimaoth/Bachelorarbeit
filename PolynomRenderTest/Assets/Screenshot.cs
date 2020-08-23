using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Screenshot : MonoBehaviour
{

    private bool takeScreenshotNextFrame = false;
    public void TakeScreenshot()
    {
        takeScreenshotNextFrame = true;
        Camera.main.targetTexture = RenderTexture.GetTemporary(Screen.width, Screen.height);
    }

    private void OnPostRender()
    {
        if (takeScreenshotNextFrame)
        {
            takeScreenshotNextFrame = false;

            var renderTexture = Camera.main.targetTexture;
            Texture2D renderResult = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.ARGB32, false);
            var rect = new Rect(0, 0, renderTexture.width, renderTexture.height);
            renderResult.ReadPixels(rect, 0, 0);

            var byteArray = renderResult.EncodeToPNG();
            File.WriteAllBytes("../screenshot.png", byteArray);

            Debug.Log("Screenshot saved");

            RenderTexture.ReleaseTemporary(renderTexture);
            Camera.main.targetTexture = null;
        }
    }
}
