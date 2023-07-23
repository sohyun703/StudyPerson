
// ToDo 항목을 추가하는 함수

function addTask() {
  const taskInput = document.getElementById("taskInput");
  const taskValue = taskInput.value.trim();
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional

  
  if (taskValue !== "") {

    const config = getFirebaseConfig();
    fetch("/add_task", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: `task=${encodeURIComponent(taskValue)}`
    })
    .then(response => response.json())
    .then(data => {
      // 성공적으로 추가되었을 때, ToDo 항목을 다시 렌더링하거나 기타 동작 수행
      console.log(data.message);
    })
    .catch(error => {
      // 오류 처리
      console.error("Error:", error);
    });
    
    taskInput.value = ""; // 입력 필드 초기화
  }


}

function getTasks() {
  fetch("/get_tasks")
    .then(response => response.json())
    .then(data => {
      const taskList = document.getElementById("taskList");
      taskList.innerHTML = "";
      data.forEach(taskData => {
        const taskItem = document.createElement("li");
        taskItem.innerText = taskData.task;
        taskList.appendChild(taskItem);
      });
    })
    .catch(error => {
      console.error("Error fetching tasks:", error);
    });
}