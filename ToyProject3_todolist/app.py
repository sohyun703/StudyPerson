# app.py
from flask import Flask, render_template, request, jsonify,send_file
import firebase_admin
from firebase_admin import credentials, db
import os

cred = credentials.Certificate("static/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://todolist-af727-default-rtdb.firebaseio.com/"
})
app = Flask(__name__, static_url_path="/static", static_folder="static")

# 나머지 Flask 앱의 코드



# 루트 경로로 접속하면 index.html 파일 렌더링

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/add_task',methods=["POST"]) #클라이언트가 해당 URL로 POST 요청을 보내면 add_task 함수가 실행된다.

def add_task():
    task = request.form.get('task') #Flask에서 클라이언트로부터 POST 요청으로 전달된 데이터에서 task라는 이름을 가지고 온다., request는 HTTP 요청과 관련된 정보를 담고 있는 객체, 클라이언트가 보낸 데이터를 얻기 위해 사용
    if task:
        try:
            new_task_ref = db.reference('tasks').push()
            new_task_ref.set({
                "task": task,
            })
            return jsonify({"message": "success"}), 200
        except Exception as e:
            return jsonify({"message": str(e)+"Internal Server Error"}), 500
    else:
        return jsonify({"message": "Task value is missing"}), 400
    

@app.route('/get_tasks', methods=["GET"])
def get_tasks():
    try:
        tasks_ref = db.reference('tasks')
        tasks_snapshot = tasks_ref.get()

        tasks_list = []
        if tasks_snapshot:
            for task_id, task_data in tasks_snapshot.items():
                tasks_list.append(task_data)

        return jsonify(tasks_list), 200
    except Exception as e:
        return jsonify({"message": str(e) + "Internal Server Error"}), 500

@app.route('/static/app.js')

def serve_js():
    return send_file('static/app.js', mimetype='text/javascript')


if __name__ == "__main__":
    app.run(debug=True)

