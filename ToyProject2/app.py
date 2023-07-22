from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# 영화 데이터를 불러올 CSV 파일명
csv_file = "movies_data.csv"

# 영화 데이터를 담을 DataFrame
movies_df = pd.read_csv(csv_file)

@app.route('/')
def index():
    return render_template('index.html', genres=list(movies_df['장르'].unique()))

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    selected_genre = request.form.get('genre')
    if selected_genre:
        recommended_movies = movies_df[movies_df['장르'] == selected_genre]
        # Extracting only the movie title and director from the DataFrame
        recommended_movies = recommended_movies[['제목', '감독']]
        return render_template('index.html', genres=list(movies_df['장르'].unique()), movies=recommended_movies.to_html(index=False))
    else:
        return render_template('index.html', genres=list(movies_df['장르'].unique()))

if __name__ == '__main__':
    app.run(debug=True)
