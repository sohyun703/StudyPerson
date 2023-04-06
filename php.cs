using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class php : MonoBehaviour
{
	void Start()
	{
		StartCoroutine("LoadFromPhp");
	}

	void Update()
	{

	}

	IEnumerator LoadFromPhp()
	{
		string url = "http://localhost/test/index.php";
		WWW www = new WWW(url);

		yield return www;

		if (www.isDone)
		{
			if (www.error == null)
			{
				Debug.Log("Receive Data : " + www.text);
			}
			else
			{
				Debug.Log("error : " + www.error);
			}
		}
	}
}

