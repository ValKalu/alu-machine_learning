-- List all records with scores from>= 10 in second_table
SELECT score, name
FROM second_table
WHERE score>= 10
ORDER BY score desc;
