-- Calculate average temperature (Fahrenheit) by city and order by temperature descending
SELECT city, ROUND(AVG(temperature), 4) AS avg_temp
FROM temperatures.SQL
GROUP BY city
ORDER BY avg_temp DESC;
