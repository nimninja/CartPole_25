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
int loopDelayMs = 60;

void moveLeft();
void moveRight();
void stopMotor();

void handleCommand(const String &cmd) {
  if (cmd == "LEFT") {
    moveLeft();
  } else if (cmd == "RIGHT") {
    moveRight();
  } else if (cmd == "STOP") {
    stopMotor();
  } else if (cmd == "MODE BALANCE") {
    loopDelayMs = 3;
    Serial.println("OK BALANCE");
  } else if (cmd == "MODE SWING") {
    loopDelayMs = 60;
    Serial.println("OK SWING");
  } else if (cmd == "RESET") {
    while (abs(beltCount) > 250) {
      if (beltCount > 0) {
        moveLeft();
      } else {
        moveRight();
      }
      delay(10);
    }
    stopMotor();
    loopDelayMs = 60;
    noInterrupts();
    beltCount = 0;
    interrupts();
    Serial.print(angleCount);
    Serial.print(",");
    Serial.println(beltCount);
  }
}

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
  // Process every pending command (mode + motor may arrive back-to-back)
  while (Serial.available()) {
    inputString = Serial.readStringUntil('\n');
    inputString.trim();
    if (inputString.length() > 0) {
      handleCommand(inputString);
    }
  }

  Serial.print(angleCount);
  Serial.print(",");
  Serial.println(beltCount);

  delay(loopDelayMs);
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
