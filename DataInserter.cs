using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class DataInserter : MonoBehaviour
{
	public string inputUserName;
	public string inputPassword;
	public string inputEmail;
	public GameObject timer;
	public float time_cur;
	string CreateUserURL = "http://localhost/test/index.php";

   
	public void call()
	{
		time_cur = timer.GetComponent<TimerExample>().time_current;
		string inputTime = time_cur.ToString();
		StartCoroutine(CreateUser(inputUserName, inputPassword, inputEmail,inputTime));
		Debug.Log("호출");

		
	}

	public IEnumerator CreateUser(string username, string password, string email,string time)
	{
		
		WWWForm form = new WWWForm();
		form.AddField("usernamePost", username);
		form.AddField("passwordPost", password);
		form.AddField("emailPost", email);
		form.AddField("timePost", time);

		UnityWebRequest www = UnityWebRequest.Post(CreateUserURL, form); // url을 텍스트로 받아오기
		yield return www.SendWebRequest();

	}
}