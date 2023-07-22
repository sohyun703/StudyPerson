import csv
import random

# Function to generate random movie data
def generate_random_movie():
    title = f"Movie {random.randint(1, 10000)}"
    genre = random.choice(["액션", "드라마", "코미디", "로맨스", "SF", "스릴러", "애니메이션"])
    director = f"Director {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"

    actors = []
    for _ in range(random.randint(1, 5)):
        actor = f"Actor {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
        actors.append(actor)

    rating = round(random.uniform(0, 10), 1)

    return [title, genre, director, ", ".join(actors), rating]

# Create 10,000 lines of data
data = [["제목", "장르", "감독", "배우", "평점"]]
for _ in range(10000):
    movie_data = generate_random_movie()
    data.append(movie_data)

# Save data to CSV file
with open("movies_data.csv", "w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print("CSV file 'movies_data.csv' with 10,000 lines of random movie data has been created.")
