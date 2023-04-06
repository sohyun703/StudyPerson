using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Timer : MonoBehaviour
{
    public TMP_Text text_Timer;
    private float time_start;
    public float time_current;
    private float time_Max = 5f;
    private bool timeActive = true;
    private void Start()
    {
        Reset_Timer();
    }
    void Update()
    {
       /* if (isEnded)
            return;

        Check_Timer();*/
    }

    public void OnTriggerEnter(Collider other)
    {
        Debug.Log("12");
        if(other.transform.tag == "cube")
        {
            StartCoroutine(Check_Timer());
           // text_Timer.text = $"{time_current:N2}";
        }
    }
    private IEnumerator Check_Timer()
    {
        time_current += Time.deltaTime;
            text_Timer.text = $"{time_current:N2}";
            Debug.Log(time_current);
            yield return null;
        

    }

    private void End_Timer()
    {
        Debug.Log("End");
        time_current = time_Max;
        text_Timer.text = $"{time_current:N2}";
      //  isEnded = true;
    }


    private void Reset_Timer()
    {
        time_start = Time.time;
        time_current = 0;
        text_Timer.text = $"{time_current:N2}";
      //  isEnded = false;
        Debug.Log("Start");
    }
}
