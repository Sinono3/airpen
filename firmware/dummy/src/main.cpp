#include <Arduino.h>
#include <Wire.h>

void scanBus(TwoWire &bus, const char *name) {
	Serial.print("Scanning ");
	Serial.print(name);
	Serial.println("...");

	bus.begin();

	bool found = false;

	for (uint8_t addr = 1; addr < 127; addr++) {
		bus.beginTransmission(addr);
		if (bus.endTransmission() == 0) {
			Serial.print(" - Device found at 0x");
			Serial.println(addr, HEX);
			found = true;
		}
	}

	if (!found) {
		Serial.println(" - No devices found");
	}

	Serial.println();
}

void setup() {
	Serial.begin(115200);
	delay(500);

	scanBus(Wire, "I2C0 (Wire)");
	scanBus(Wire1, "I2C1 (Wire1)");
}

void loop() {}
