#include <PinChangeInterrupt.h>
#include <PinChangeInterruptBoards.h>
#include <PinChangeInterruptPins.h>
#include <PinChangeInterruptSettings.h>

#include <PinChangeInterrupt.h>

// === Pins ===
const int anglePinA = 4;
const int anglePinB = 5;
const int beltPinA  = 2;
const int beltPinB  = 3;

const int motorPin1 = 13;
const int motorPin2 = 12;

// === Encoder counts ===
volatile long angleCount = 0;
volatile long beltCount = 0;

String inputString = "";

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for Serial to open

  pinMode(anglePinA, INPUT_PULLUP);
  pinMode(anglePinB, INPUT_PULLUP);
  pinMode(beltPinA, INPUT_PULLUP);
  pinMode(beltPinB, INPUT_PULLUP);

  attachPCINT(digitalPinToPCINT(anglePinA), angleISR, RISING);
  attachPCINT(digitalPinToPCINT(anglePinB), angleISR_B, RISING);
  attachInterrupt(digitalPinToInterrupt(beltPinA), beltISR, RISING);
  attachInterrupt(digitalPinToInterrupt(beltPinB), beltISR_B, RISING);


  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  stopMotor();
}

void loop() {
  // === Handle commands from Python ===
  if (Serial.available()) {
    inputString = Serial.readStringUntil('\n');
    inputString.trim();

    if (inputString == "LEFT") {
      moveLeft();
    } else if (inputString == "RIGHT") {
      moveRight();
    } else if (inputString == "STOP") {
      stopMotor();
    } else if (inputString == "RESET") {
      stopMotor();
      while (abs(beltCount) > 250) {
        if (beltCount > 0) {
          moveLeft();
        } else {
          moveRight();
        }
        delay(10);
      }
      stopMotor();
      // Logical origin: cart center, pole hanging (count 0 = hang, 600 = upright)
      noInterrupts();
      beltCount = 0;
      angleCount = 0;
      interrupts();
      Serial.print(angleCount);
      Serial.print(",");
      Serial.println(beltCount);
      return;  // skip extra print this loop iteration
    }
  }

  if (abs(angleCount) >= 1200) {
    angleCount = 0;
  }

  // === Send encoder data to Python ===
  Serial.print(angleCount);
  Serial.print(",");
  Serial.println(beltCount);


  delay(60);
}

// === Motor control ===
void moveLeft() {
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
}

void moveRight() {
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
}

void stopMotor() {
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
}

// === Interrupts for encoders ===
void angleISR() {
  if (digitalRead(anglePinB) == LOW)
    angleCount++;
  else
    angleCount--;
}

void angleISR_B() {
  if (digitalRead(anglePinA) == LOW)
    angleCount--;
  else
    angleCount++;
}

void beltISR() {
  if (digitalRead(beltPinB) == LOW)
    beltCount++;
  else
    beltCount--;
}       

void beltISR_B() {
  if (digitalRead(beltPinA) == LOW)
    beltCount--;
  else
    beltCount++;
}
