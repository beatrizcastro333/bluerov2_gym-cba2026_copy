#include <Arduino.h>

// ---------------------------------------------------------
// 1. DEFINIÇÃO DO CONTROLADOR PID
// ---------------------------------------------------------
class PID6DoF {
  private:
    float kp[6], ki[6], kd[6];
    float dt;
    float integral[6], prev_error[6];
    bool first_step;

  public:
    PID6DoF(float delta_t) {
        dt = delta_t;
        reset();
    }

    void setGains(float p[6], float i[6], float d[6]) {
        for(int j = 0; j < 6; j++) {
            kp[j] = p[j];
            ki[j] = i[j];
            kd[j] = d[j];
        }
    }

    void reset() {
        for(int j = 0; j < 6; j++) {
            integral[j] = 0.0f;
            prev_error[j] = 0.0f;
        }
        first_step = true;
    }

    void compute(float error[6], float u[6]) {
        for(int j = 0; j < 6; j++) {
            integral[j] += error[j] * dt;
            if (first_step) {
                prev_error[j] = error[j];
            }
            
            float derivative = (error[j] - prev_error[j]) / dt;
            prev_error[j] = error[j];
            u[j] = (kp[j] * error[j]) + (ki[j] * integral[j]) + (kd[j] * derivative);
        }
        first_step = false;

        // Limite de força/torque (N ou Nm) por eixo
        u[0] = constrain(u[0], -40.0f, 40.0f); // Surge (N)
        u[1] = constrain(u[1], -40.0f, 40.0f); // Sway (N)
        u[2] = constrain(u[2], -40.0f, 40.0f); // Heave (N)
        u[3] = 0.0f; // Roll (Nm)
        u[4] = 0.0f; // Pitch (Nm)
        u[5] = constrain(u[5], -5.0f, 5.0f); // Yaw (Nm)
    }
};

// Instancia o controlador
PID6DoF meuPid(0.1f);

void setup() {
    Serial.begin(115200);
}

void loop() {
    if (Serial.available() > 0) {
        String msg = Serial.readStringUntil('\n');
        
        char buf[256];
        msg.toCharArray(buf, sizeof(buf));
        char* token = strtok(buf, ",");

        if (token != NULL) {
            // G = Atualiza os 18 ganhos (Kp[6], Ki[6], Kd[6])
            if (token[0] == 'G') {
                float p[6], i[6], d[6];
                for(int j = 0; j < 6; j++) p[j] = atof(strtok(NULL, ","));
                for(int j = 0; j < 6; j++) i[j] = atof(strtok(NULL, ","));
                for(int j = 0; j < 6; j++) d[j] = atof(strtok(NULL, ","));
                
                meuPid.setGains(p, i, d);
                meuPid.reset();
                Serial.println("GANHOS_OK");
                
            } 
            // E = Recebe o erro em referencial de corpo e calcula ação de controle
            else if (token[0] == 'E') {
                float error_body[6];
                for(int j = 0; j < 6; j++) {
                    error_body[j] = atof(strtok(NULL, ","));
                }
                
                float action_6dof[6];
                meuPid.compute(error_body, action_6dof);
                
                // Retorna as 6 ações calculadas
                Serial.print("U,");
                for(int j = 0; j < 5; j++) {
                    Serial.print(action_6dof[j], 6);
                    Serial.print(",");
                }
                Serial.println(action_6dof[5], 6);
            }
        }
    }
}
