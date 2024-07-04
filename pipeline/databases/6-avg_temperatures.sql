-- Import hbtn_0c_0 database table temperatures.sql
-- Calculate average temperature (Fahrenheit) by city, ordered by temperature (descending)
SELECT city, AVG(temperature) AS average_temperature_fahrenheit
FROM temperatures
GROUP BY city
ORDER BY average_temperature_fahrenheit DESC;
