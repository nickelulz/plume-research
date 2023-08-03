#include <Arduino.h>
#include "DHT.h"
#define DHTPIN 4
#define DHTTYPE DHT11

#define ERROR_PIN 10
#define SUCCESS_PIN 13

DHT dht(DHTPIN, DHTTYPE);

void setup() {
	Serial.begin(9600);

	// initialize the sensor
	dht.begin();

	pinMode(ERROR_PIN, OUTPUT);
	pinMode(SUCCESS_PIN, OUTPUT);
}

void loop() 
{
	if (Serial.available() > 0) {
		String data = Serial.readStringUntil('\n');

		if (data.equals("GEN_WEATHER_DATA")) 
		{
			float humidity = dht.readHumidity();
			// taken in Celsius
			float temperature = dht.readTemperature();

			// Check for input failure
			if (isnan(humi) || isnan(tempC) || isnan(tempF)) {
				Serial.println("DHT ERROR");

				// Light error led on failure
				digitalWrite(ERROR_PIN, HIGH);
			}
			else {
				Serial.print(humidity);
				Serial.print(",");
				Serial.println(temperature);

				// Flash success led

				// on
			    digitalWrite(SUCCESS_PIN, HIGH);
			    delay(500);

			    // off
			    digitalWrite(SUCCESS_PIN, LOW);
			    delay(500);
			}
		}
	}
	delay(2000);
}
