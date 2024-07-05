---A Script that displays max tempreature of each state
SELECT state, max(temperature) as max_temp
FROM temperatures
GROUP BY state
ORDER BY state;
