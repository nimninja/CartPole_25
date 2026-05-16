#include <PinChangeInterrupt.h>
#include <PinChangeInterruptBoards.h>
#include <PinChangeInterruptPins.h>
#include <PinChangeInterruptSettings.h>

#include <PinChangeInterrupt.h>
#include <math.h>

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

// Motor command in [-1, 1]: sign = direction, magnitude = speed
float motorCmd = 0.0f;
const float MOTOR_DEADZONE = 0.05f;
const unsigned int PWM_PERIOD_US = 2000;

void serviceMotorPwm();
void moveLeft();
void moveRight();
void stopMotor();
void angleISR();
void angleISR_B();
void beltISR();
void beltISR_B();

void setup() {
  Serial.begin(9600);
  while (!Serial);

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
  if (Serial.available()) {
    inputString = Serial.readStringUntil('\n');
    inputString.trim();

    if (inputString == "RESET") {
      motorCmd = 0.0f;
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
      noInterrupts();
      beltCount = 0;
      interrupts();
      Serial.print(angleCount);
      Serial.print(",");
      Serial.println(beltCount);
    } else if (inputString == "STOP") {
      motorCmd = 0.0f;
      stopMotor();
    } else if (inputString == "LEFT") {
      motorCmd = -1.0f;
    } else if (inputString == "RIGHT") {
      motorCmd = 1.0f;
    } else {
      float cmd = inputString.toFloat();
      if (inputString.length() > 0) {
        motorCmd = constrain(cmd, -1.0f, 1.0f);
      }
    }
  }

  serviceMotorPwm();

  if (abs(angleCount) >= 1200) {
    angleCount = 0;
  }

  Serial.print(angleCount);
  Serial.print(",");
  Serial.println(beltCount);

  delay(60);
}

void serviceMotorPwm() {
  if (fabs(motorCmd) < MOTOR_DEADZONE) {
    stopMotor();
    return;
  }
  int duty = (int)(fabs(motorCmd) * 255.0f);
  unsigned long phase = micros() % PWM_PERIOD_US;
  unsigned long onTime = (unsigned long)duty * PWM_PERIOD_US / 255UL;
  if (phase < onTime) {
    if (motorCmd > 0) {
      moveRight();
    } else {
      moveLeft();
    }
  } else {
    stopMotor();
  }
}

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
