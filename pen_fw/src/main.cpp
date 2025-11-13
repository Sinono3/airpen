#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// A flag to track whether we should be recording data.
bool isRecording = false;

// A variable to store the exact time when recording starts.
unsigned long recordingStartTime;

int counter = 0;

// --- NEW: A variable to track the state of BUTTON_3 for single-press detection ---
int lastButton3State = HIGH;


// The setup function runs once when you press reset or power the board on.
void setup() {
  // Initialize Serial communication at 115200 bits per second.
  Serial.begin(115200);

  // Configure the built-in LED as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW); // Ensure the LED is off initially.

  // Configure Button A as an input with an internal pull-up resistor.
  pinMode(BUTTON_1, INPUT_PULLUP);

  // --- NEW: Configure BUTTON_3 as an input with an internal pull-up resistor. ---
  pinMode(BUTTON_3, INPUT_PULLUP);

  // --- Initialize the MPU6050 Sensor ---
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    // Halt execution if the sensor isn't found.
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
    }
  }
  Serial.println("MPU6050 Found!");

  // --- (Optional but Recommended) Set sensor ranges ---
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

// The loop function runs over and over again forever.
void loop() {
      // 1. Light up the LED to show we're recording.
      digitalWrite(LED_BUILTIN, HIGH);

      // 2. Read sensor data
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);

      // Convert gyro from rad/s to deg/s
      float gyroX = g.gyro.x * 180 / M_PI;
      float gyroY = g.gyro.y * 180 / M_PI;
      float gyroZ = g.gyro.z * 180 / M_PI;
      
      // 3. Print data in CSV format
      Serial.print(a.acceleration.x);
      Serial.print(",");
      Serial.print(a.acceleration.y);
      Serial.print(",");
      Serial.print(a.acceleration.z);
      Serial.print(",");
      Serial.print(gyroX);
      Serial.print(",");
      Serial.print(gyroY);
      Serial.print(",");
      Serial.print(gyroZ);
      Serial.print("\n");

      // 4. Wait for a short period to control the data sampling rate.
      delay(10);
}
