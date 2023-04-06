using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class start : MonoBehaviour
{
    public GameObject obj;
    private void OnTriggerEnter(Collider other)
    {

        if (other.transform.tag == "director")
        {
            obj.GetComponent<TimerExample>().start();
        }
    }
}
