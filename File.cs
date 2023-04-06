using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
public class File : MonoBehaviour
{
    // Start is called before the first frame update
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.W))
        {
            string filePath = Path.Combine(Application.streamingAssetsPath, "Example.txt");
            string message = "I love me !!";

            WriteTxt(filePath, message);
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            string filePath = Path.Combine(Application.streamingAssetsPath, "Example.txt");
            Debug.Log(ReadTxt(filePath));
        }
    }

    //텍스트 쓰기 함수
    void WriteTxt(string filePath, string message)
    {
        DirectoryInfo directoryInfo = new DirectoryInfo(Path.GetDirectoryName(filePath));

        if (!directoryInfo.Exists)
        {
            directoryInfo.Create();
        }

        FileStream fileStream
            = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.Write);

        StreamWriter writer = new StreamWriter(fileStream, System.Text.Encoding.Unicode);

        writer.WriteLine(message);
        writer.Close();
    }

    //텍스트 읽기 함수
    string ReadTxt(string filePath)
    {
        FileInfo fileInfo = new FileInfo(filePath);
        string value = "";

        if (fileInfo.Exists)
        {
            StreamReader reader = new StreamReader(filePath);
            value = reader.ReadToEnd();
            reader.Close();
        }

        else
            value = "파일이 없습니다.";

        return value;
    }
}

