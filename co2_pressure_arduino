#define LedPin 13
#define pwmPin 5

int SensorPin = A0;

int prevVal = LOW;

long th, tl, h, l, ppm;


 

void setup() {

  Serial.begin (9600);

  delay(1000);

  pinMode(pwmPin, INPUT);

  pinMode(LedPin, OUTPUT);

}

 
void showco2(){

  long tt = millis();

  int myVal = digitalRead(pwmPin);

  if (myVal == HIGH) {

    digitalWrite(LedPin, HIGH);

    if (myVal != prevVal) {

      h = tt;

      tl = h-l;

      prevVal = myVal;

    }

  }else{

    digitalWrite(LedPin, LOW);

    if (myVal != prevVal) {

      l = tt;

      th = l-h;

      prevVal = myVal;

      ppm = 5000 * (th-2) / (th + tl-4);

      Serial.println(int(ppm));


      showfsr();

    }

  }

}


void showfsr(){

  int SensorReading = analogRead(SensorPin);

  int mfsr_r18 = map(SensorReading, 0, 1024, 0, 255);

  Serial.println(mfsr_r18);

}


void loop() {

  //showfsr(); 

  showco2();

  //showco2();


}
