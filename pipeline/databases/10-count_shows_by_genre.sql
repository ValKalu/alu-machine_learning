-- Script that shows number of shows by genre
SELECT genre AS genre, COUNT(*) AS number_of_shows
FROM shows
GROUP BY genre 
ORDER BY number_of_shows DESC;
