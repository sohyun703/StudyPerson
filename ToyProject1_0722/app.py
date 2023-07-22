from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app =Flask(__name__)

# 한글 폰트 설정
plt.rc('font', family='NanumGothic')

data = {
    '이름': ['홍길동', '김철수', '이영희', '박영수'],
    '나이': [25, 30, 28, 22],
    '성별': ['남', '남', '여', '남'],
    '점수': [90, 85, 78, 92]
}

@app.route('/')

def index():
    return render_template('index.html',)

@app.route('/data')

def data_analysis():
    df = pd.DataFrame(data)
    avg_score = df['점수'].mean()
    max_score=df['점수'].max()
    min_score =df['점수'].min()
    
    return render_template('index.html',avg_score=avg_score, max_score=max_score, min_score=min_score)

@app.route('/show')

def data_visualize():
    df= pd.DataFrame(data)
    name_list = df['이름'].tolist()
    x_pos = np.arange(len(name_list))
    score_list= df['점수'].tolist()
    y_pos = np.arange(len(score_list))
    
    plt.bar(x_pos,y_pos,align='center')
    plt.ylabel('성적')
    plt.title('학생 성적 그래프')
    
    img = io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return render_template('index.html',plot_url=plot_url)

if __name__ =='__main__':
    app.run(debug=True)