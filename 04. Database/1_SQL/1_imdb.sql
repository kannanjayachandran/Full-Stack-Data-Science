-- Selecting the database
\c imdb

-- show tables
\dt

-- show table schema
\d+ imdb


SELECT DISTINCT first_name, last_name FROM director ORDER BY first_name;

SELECT name, year, rank_score FROM movies WHERE rank_score>9.0 ORDER BY rank_score DESC LIMIT 10;

SELECT name, year, rank_score FROM movies WHERE year BETWEEN 2000 AND 2002 ORDER BY rank_score DESC;

SELECT director_id, genre FROM movies WHERE genre IN ('Horror', 'Action', 'Sci-Fi') ORDER BY genre;

SELECT COUNT(*) FROM movies WHERE year BETWEEN 2000 AND 2010;

SELECT year, COUNT(year) FROM movies GROUP BY year ORDER BY year;

SELECT year, COUNT(year) year_count FROM movies GROUP BY year HAVING year_count>500 ORDER BY year_count DESC;

SELECT m.movies, g.genre FROM movies m JOIN movie_genres g ON m.id=g.movie_id LIMIT 30; 

SELECT a.first_name, a.last_name FROM actors a JOIN roles r ON a.id=r.actor_id JOIN movies m ON m.id=r.movie_id AND m.name='The Dark Knight';

SELECT first_name, last_name FROM actors WHERE id IN 
    (SELECT actor_id from roles WHERE movie_id IN)
        (SELECT id FROM movies WHERE name='The Dark Knight');