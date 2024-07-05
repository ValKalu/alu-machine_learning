-- Script that list all shows as tv_show,title-tv_show.genre_id
SELECT tv_shows.title, tvshows_genres.genres_id
FROM hbtn_0d_tv_shows
LEFT JOIN tv_shows_genres ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY tv_shows.title;