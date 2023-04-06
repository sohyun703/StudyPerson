using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
public class end : MonoBehaviour
{
   

    public GameObject obj;
    public GameObject datainsert;
    private void OnTriggerEnter(Collider other)
    {

        if (other.transform.tag == "director")
        {
            obj.GetComponent<TimerExample>().stop();
            datainsert.GetComponent<DataInserter>().call();
        }
    }


}
