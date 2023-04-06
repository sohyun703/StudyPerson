using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
public class TimerExample : MonoBehaviour
{
    public TMP_Text text_Timer;
    public float time_current;
    bool str;
    // Start is called before the first frame update
    void Start()
    {
        time_current = 0;
        str = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (str)
        {
            time_current += Time.deltaTime;
        }
       // text_Timer.text = $"{val:N2}";
        text_Timer.text = time_current.ToString();
    }

    public void start()
    {
        str = true;
    }
    public void stop()
    {
        str = false;
    }
    public void reset()
    {
        str = false;
        time_current = 0;
    }
}
