#include <Arduino.h>
#include <MPU9250_asukiaaa.h>

MPU9250_asukiaaa mySensor;
float aX, aY, aZ, aSqrt;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  mySensor.setWire(&Wire);
  mySensor.beginAccel();
}

void loop() {
  mySensor.accelUpdate();
  aX = mySensor.accelX();
  aY = mySensor.accelY();
  aZ = mySensor.accelZ();
  aSqrt = mySensor.accelSqrt();

  Serial.print("aX=");
  Serial.print(aX);
  Serial.print("aY=");
  Serial.print(aY);
  Serial.print("aZ=");
  Serial.print(aZ);
  Serial.print("aSqrt=");
  Serial.print(aSqrt);
  Serial.println();
}
