import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import serial
from scipy.optimize import minimize

# Importando modelos dinâmicos
from dynamics_calm import DynamicsCalm
from dynamics_storm import DynamicsStorm

# Comunicação Serial
PORTA_SERIAL = 'COM6'  ###
BAUD_RATE = 115200

try:
    ser = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=0.1)
    time.sleep(2.0)  # estabilização
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print(f"[OK] Conectado com sucesso na porta {PORTA_SERIAL}!")
except Exception as e:
    print(f"[ERRO] Não foi possível abrir a porta {PORTA_SERIAL}: {e}")
    ser = None


def enviar_ganhos_stm32(kp, ki, kd, nome_cenario=""):
    # Envia os ganhos otimizados para o STM32
    if ser is None or not ser.is_open:
        print("[ERRO] Porta serial desconectada.")
        return

    print(f"\n--- ENVIANDO GANHOS OTIMIZADOS PARA STM32 [{nome_cenario.upper()}] ---")
    print(f"Kp: {np.round(kp, 4)}")
    print(f"Ki: {np.round(ki, 4)}")
    print(f"Kd: {np.round(kd, 4)}")
    print("-" * 50)

    valores = list(kp) + list(ki) + list(kd)
    msg = "G," + ",".join(f"{v:.6f}" for v in valores) + "\n"

    ser.reset_input_buffer()
    ser.write(msg.encode('utf-8'))
    time.sleep(0.1)

    resposta = ser.readline().decode('utf-8').strip()
    print(f"[STM32 Resp]: {resposta}\n")


def calcular_acao_stm32(error_body):
    # Envia o erro **6DoF para o STM32 e recebe comandos de controle
    if ser is None or not ser.is_open:
        return np.zeros(6)

    msg = "E," + ",".join(f"{v:.6f}" for v in error_body) + "\n"
    ser.write(msg.encode('utf-8'))

    linha = ser.readline().decode('utf-8').strip()

    if linha.startswith("U,"):
        linha = linha[2:]

    partes = linha.split(",")
    if len(partes) == 6:
        try:
            return np.array([float(p) for p in partes])
        except ValueError:
            pass

    return np.zeros(6)


# ---------------------------------------------------------
# 1. DEFINIÇÃO DO CONTROLADOR PID
# ---------------------------------------------------------
class PID6DoF:
    def __init__(self, kp, ki, kd, dt):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt

        self.integral = np.zeros(6) ###
        self.prev_error = np.zeros(6) ###
        self.first_step = True

    def compute(self, error):
        self.integral += error * self.dt

        if self.first_step:
            self.prev_error = error.copy()
            self.first_step = False

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Limite de força/torque (N ou Nm) por eixo
        u[0] = np.clip(u[0], -40.0, 40.0) # Surge (N)
        u[1] = np.clip(u[1], -40.0, 40.0) # Sway (N)
        u[2] = np.clip(u[2], -40.0, 40.0) # Heave (N)
        u[3] = 0.0 # Roll (Nm)
        u[4] = 0.0 # Pitch (Nm)
        u[5] = np.clip(u[5], -5.0, 5.0) # Yaw (Nm)
        
        return u

    def reset(self):
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.first_step = True


# Objective Function - otimização
### CALM
def objective_function_calm(gains):
    model = DynamicsCalm()
    model.SEA_STATE = "calm"

    kp_opt = gains[0:6]
    ki_opt = gains[6:12]
    kd_opt = gains[12:18]

    # Treinando c/ dois extremos
    alvos_treino = [
        np.array([1.5, 0.5, 1.0, 0.5, 0.4, 0.2]),
        np.array([-1.5, -0.5, -1.0, -0.5, -0.4, -0.2])
    ]

    custo_total = 0.0
    lambda_u = 0.001

    for target in alvos_treino:
        pid = PID6DoF(kp_opt, ki_opt, kd_opt, model.dt)
        state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'u': 0.0, 'v': 0.0, 'w': 0.0,
            'p': 0.0, 'q': 0.0, 'r': 0.0
        }

        steps = int(10.0 / model.dt) ###
        j_error = 0.0
        j_energy = 0.0

        ###
        for _ in range(steps):
            curr_pos = np.array([state['x'], state['y'], state['z'],
                                 state['roll'], state['pitch'], state['yaw']])
            error_global = target - curr_pos

            for i in range(3, 6):
                error_global[i] = (error_global[i] + np.pi) % (2 * np.pi) - np.pi

            phi = state["roll"]
            theta = state["pitch"]
            psi = state["yaw"]

            # Senos e cossenos
            c_phi, s_phi = np.cos(phi), np.sin(phi)
            c_theta, s_theta = np.cos(theta), np.sin(theta)
            c_psi, s_psi = np.cos(psi), np.sin(psi)

            # matrizes de rotação para cada eixo
            R_x = np.array([[1, 0, 0], [0, c_phi, -s_phi], [0, s_phi, c_phi]])
            R_y = np.array([[c_theta, 0, s_theta], [0, 1, 0], [-s_theta, 0, c_theta]])
            R_z = np.array([[c_psi, -s_psi, 0], [s_psi, c_psi, 0], [0, 0, 1]])

            R_nb = R_z @ R_y @ R_x
            R_bn = R_nb.T

            # Erro global para o referencial do corpo
            error_body = np.zeros(6)

            # Para os 3 eixos de translação
            error_body[0:3] = R_bn @ error_global[0:3]

            # Para os 3 eixos de rotação
            error_body[3:6] = error_global[3:6]

            tau_pid = pid.compute(error_body)

            action_6dof = tau_pid
            ###

            model.step(state, action_6dof)

            j_error += np.sum(error_global**2) * model.dt
            j_energy += np.sum(tau_pid**2) * model.dt

        custo_total += (j_error + lambda_u * j_energy)

        ###
        if np.isnan(j_error) or np.isinf(j_error):
            return 1e9
        ###

    return custo_total

### STORM
def objective_function_storm(gains):
    model = DynamicsStorm()
    model.SEA_STATE = "storm"

    kp_opt = gains[0:6] ###
    ki_opt = gains[6:12] ###
    kd_opt = gains[12:18] ###

    # Treinando c/ dois extremos
    alvos_treino = [
        np.array([1.5, 0.5, 1.0, 0.0, 0.0, 0.2]),
        np.array([-1.5, -0.5, -1.0, 0.0, 0.0, -0.2])
    ]

    custo_total = 0.0
    lambda_u = 0.001

    for target in alvos_treino:
        pid = PID6DoF(kp_opt, ki_opt, kd_opt, model.dt)
        state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'u': 0.0, 'v': 0.0, 'w': 0.0,
            'p': 0.0, 'q': 0.0, 'r': 0.0
        }

        steps = int(10.0 / model.dt) ###
        j_error = 0.0
        j_energy = 0.0

        ###
        for _ in range(steps):
            curr_pos = np.array([state['x'], state['y'], state['z'],
                                 state['roll'], state['pitch'], state['yaw']])
            error_global = target - curr_pos
            for i in range(3, 6):
                error_global[i] = (error_global[i] + np.pi) % (2 * np.pi) - np.pi

            phi = state["roll"]
            theta = state["pitch"]
            psi = state["yaw"]

            # Senos e cossenos
            c_phi, s_phi = np.cos(phi), np.sin(phi)
            c_theta, s_theta = np.cos(theta), np.sin(theta)
            c_psi, s_psi = np.cos(psi), np.sin(psi)

            # matrizes de rotação para cada eixo
            R_x = np.array([[1, 0, 0], [0, c_phi, -s_phi], [0, s_phi, c_phi]])
            R_y = np.array([[c_theta, 0, s_theta], [0, 1, 0], [-s_theta, 0, c_theta]])
            R_z = np.array([[c_psi, -s_psi, 0], [s_psi, c_psi, 0], [0, 0, 1]])

            R_nb = R_z @ R_y @ R_x
            R_bn = R_nb.T

            # Erro global para o referencial do corpo
            error_body = np.zeros(6)

            # Para os 3 eixos de translação
            error_body[0:3] = R_bn @ error_global[0:3]

            # Para os 3 eixos de rotação
            error_body[3:6] = error_global[3:6]

            tau_pid = pid.compute(error_body)

            action_6dof = tau_pid
            ###

            model.step(state, action_6dof)

            j_error += np.sum(error_global**2) * model.dt
            j_energy += np.sum(tau_pid**2) * model.dt

        custo_total += (j_error + lambda_u * j_energy)

        ###
        if np.isnan(j_error) or np.isinf(j_error):
            return 1e9
        ###

    return custo_total


# Run Experiment
### CALM
def run_experiment_calm(kp, ki, kd, target_pos=(1.5, 0.5, 1.0, 0.0, 0.0, 0.2)):
    model = DynamicsCalm()
    model.SEA_STATE = "calm"

    # Estado inicial: parado na origem
    state = {
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'u': 0.0, 'v': 0.0, 'w': 0.0,
        'p': 0.0, 'q': 0.0, 'r': 0.0
    }

    # Alvos
    target = np.array(target_pos)

    history = []
    total_time = 40.0
    steps = int(total_time / model.dt)

    ###
    for i in range(steps):
        curr_pos = np.array([state['x'], state['y'], state['z'],
                             state['roll'], state['pitch'], state['yaw']])
        error_global = target - curr_pos

        for j in range(3, 6):
            error_global[j] = (error_global[j] + np.pi) % (2 * np.pi) - np.pi

        phi = state["roll"]
        theta = state["pitch"]
        psi = state["yaw"]

        # Senos e cossenos
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        # matrizes de rotação para cada eixo
        R_x = np.array([[1, 0, 0], [0, c_phi, -s_phi], [0, s_phi, c_phi]])
        R_y = np.array([[c_theta, 0, s_theta], [0, 1, 0], [-s_theta, 0, c_theta]])
        R_z = np.array([[c_psi, -s_psi, 0], [s_psi, c_psi, 0], [0, 0, 1]])

        R_nb = R_z @ R_y @ R_x
        R_bn = R_nb.T

        # Erro global para o referencial do corpo
        error_body = np.zeros(6)

        # Para os 3 eixos de translação
        error_body[0:3] = R_bn @ error_global[0:3]

        # Para os 3 eixos de rotação
        error_body[3:6] = error_global[3:6]

        tau_pid = calcular_acao_stm32(error_body)

        action_6dof = tau_pid
        ###

        model.step(state, action_6dof)

        history.append({
            'time': i * model.dt,
            'x': state['x'], 'y': state['y'], 'z': state['z'],
            'roll': np.degrees(state['roll']), ###
            'pitch': np.degrees(state['pitch']), ###
            'yaw': np.degrees(state['yaw']),
            'error_pos_norm': np.linalg.norm(error_global[:3]),
            'error_ori_norm': np.linalg.norm(error_global[3:6]) * 180 / np.pi,
            'error_roll': np.abs(error_global[3] * 180 / np.pi),
            'error_pitch': np.abs(error_global[4] * 180 / np.pi),
            'error_yaw': np.abs(error_global[5] * 180 / np.pi),
            'tau_surge': tau_pid[0],
            'tau_sway': tau_pid[1],
            'tau_heave': tau_pid[2], ###
            'tau_roll': tau_pid[3], ###
            'tau_pitch': tau_pid[4], ###
            'tau_yaw': tau_pid[5], ###
            'esforco_total': np.linalg.norm(tau_pid) ###
        })

    return pd.DataFrame(history), target


### STORM
def run_experiment_storm(kp, ki, kd, target_pos=(1.5, 0.5, 1.0, 0.0, 0.0, 0.2)):
    model = DynamicsStorm()
    model.SEA_STATE = "storm"

    # Estado inicial: parado na origem
    state = {
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'u': 0.0, 'v': 0.0, 'w': 0.0,
        'p': 0.0, 'q': 0.0, 'r': 0.0
    }

    # Alvos
    target = np.array(target_pos)

    history = []
    total_time = 40.0
    steps = int(total_time / model.dt)

    ###
    for i in range(steps):
        curr_pos = np.array([state['x'], state['y'], state['z'],
                             state['roll'], state['pitch'], state['yaw']])
        error_global = target - curr_pos

        for j in range(3, 6):
            error_global[j] = (error_global[j] + np.pi) % (2 * np.pi) - np.pi

        phi = state["roll"]
        theta = state["pitch"]
        psi = state["yaw"]

        # Senos e cossenos
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        # matrizes de rotação para cada eixo
        R_x = np.array([[1, 0, 0], [0, c_phi, -s_phi], [0, s_phi, c_phi]])
        R_y = np.array([[c_theta, 0, s_theta], [0, 1, 0], [-s_theta, 0, c_theta]])
        R_z = np.array([[c_psi, -s_psi, 0], [s_psi, c_psi, 0], [0, 0, 1]])

        R_nb = R_z @ R_y @ R_x
        R_bn = R_nb.T

        # Erro global para o referencial do corpo
        error_body = np.zeros(6)

        # Para os 3 eixos de translação
        error_body[0:3] = R_bn @ error_global[0:3]

        # Para os 3 eixos de rotação
        error_body[3:6] = error_global[3:6]

        tau_pid = calcular_acao_stm32(error_body)

        action_6dof = tau_pid
        ###

        model.step(state, action_6dof)

        history.append({
            'time': i * model.dt,
            'x': state['x'], 'y': state['y'], 'z': state['z'],
            'roll': np.degrees(state['roll']), ###
            'pitch': np.degrees(state['pitch']), ###
            'yaw': np.degrees(state['yaw']),
            'error_pos_norm': np.linalg.norm(error_global[:3]),
            'error_ori_norm': np.linalg.norm(error_global[3:6]) * 180 / np.pi,
            'error_roll': np.abs(error_global[3] * 180 / np.pi),
            'error_pitch': np.abs(error_global[4] * 180 / np.pi),
            'error_yaw': np.abs(error_global[5] * 180 / np.pi),
            'tau_surge': tau_pid[0],
            'tau_sway': tau_pid[1],
            'tau_heave': tau_pid[2], ###
            'tau_roll': tau_pid[3], ###
            'tau_pitch': tau_pid[4], ###
            'tau_yaw': tau_pid[5], ###
            'esforco_total': np.linalg.norm(tau_pid) ###
        })

    return pd.DataFrame(history), target


# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":

    # Otimização dos Ganhos
    # Ganhos iniciais [Kp, Ki, Kd] para os 4 eixos
    x0 = [15.0, 15.0, 15.0, 5.0, 5.0, 5.0,    # Kp inicial
          0.1, 0.1, 0.1, 0.02, 0.02, 0.02,   # Ki inicial
          4.0, 4.0, 4.0, 1.0, 1.0, 1.0]     # Kd inicial

    ### CALM
    # Limites de busca
    bounds_calm = [(0, 50)]*6 + [(0, 5)]*6 + [(0, 80)]*6

    print("Iniciando otimização L-BFGS-B (CALM)... Isso pode levar alguns segundos.")
    res_calm = minimize(objective_function_calm, x0, bounds=bounds_calm, method='L-BFGS-B', options={'maxiter': 50, 'disp': True})

    best_kp_calm = res_calm.x[0:6] ###
    best_ki_calm = res_calm.x[6:12] ###
    best_kd_calm = res_calm.x[12:18] ###

    print("\n===== GANHOS OTIMIZADOS - CALM =====")
    print(f"Melhores Kp: {best_kp_calm.tolist()}")
    print(f"Melhores Ki: {best_ki_calm.tolist()}")
    print(f"Melhores Kd: {best_kd_calm.tolist()}")

    ### STORM
    # Limites de busca
    bounds_storm = [(0, 40)]*6 + [(0, 5)]*6 + [(0, 15)]*6
    print("\nIniciando otimização L-BFGS-B (STORM)... Isso pode levar alguns segundos.")

    res_storm = minimize(objective_function_storm, x0, bounds=bounds_storm, method='L-BFGS-B', options={'maxiter': 50, 'disp': True})

    best_kp_storm = res_storm.x[0:6] ###
    best_ki_storm = res_storm.x[6:12] ###
    best_kd_storm = res_storm.x[12:18] ###

    print("\n===== GANHOS OTIMIZADOS - STORM =====")
    print(f"Melhores Kp: {best_kp_storm.tolist()}")
    print(f"Melhores Ki: {best_ki_storm.tolist()}")
    print(f"Melhores Kd: {best_kd_storm.tolist()}")


    # Alvos para a simulação
    ten_targets = [
        (1.0, 1.5, 1.0, 0.0, 0.0, 1.0),
        (1.2, -0.4, 1.5, 0.0, 0.0, 0.4),
        (-1.0, 0.8, 2.0, 0.0, 0.0, -0.3),
        (0.8, 1.2, 1.2, 0.0, 0.0, 0.4),
        (-1.5, -0.5, 2.5, 0.0, 0.0, -0.9),
        (1.0, 1.0, 0.5, 0.0, 0.0, 0.8),
        (-0.5, 1.5, 1.8, 0.0, 0.0, 0.5),
        (0.0, 1.5, 1.0, 0.0, 0.0, -0.4),
        (1.4, 0.0, 2.2, 0.0, 0.0, 0.4),
        (-1.2, 0.0, 1.5, 0.0, 0.0, -0.6)
    ]

    # --- Configurações de estilo e utilidades ---
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--"
    })
    
    OUTPUT_DIR = "plots_stationkeeping_hil"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_plot(name):
        # Função p/ salvar os gráficos em PDF e PNG
        pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
        png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        print(f" [OK] Saved: {pdf_path}")


    # SIMULAÇÕES HIL
    ### CALM
    print("\n--- CALM ---\n")
    enviar_ganhos_stm32(best_kp_calm, best_ki_calm, best_kd_calm, nome_cenario="CALM")

    erros_historico_otimizado_calm = []
    erros_historico_ori_otimizado_calm = []
    df_first_calm = None

    for i, alvo in enumerate(ten_targets):
        print(f"Executando alvo CALM HIL {i+1}/{len(ten_targets)}...")
        df_target, _ = run_experiment_calm(best_kp_calm, best_ki_calm, best_kd_calm, target_pos=alvo)
        if i == 0:
            df_first_calm = df_target
        erros_historico_otimizado_calm.append(df_target['error_pos_norm'].values)
        erros_historico_ori_otimizado_calm.append(df_target['error_ori_norm'].values)

    time_vector_calm = df_first_calm['time']

    erros_matriz_otimizado_calm = np.array(erros_historico_otimizado_calm)
    erro_medio_otimizado_calm = np.mean(erros_matriz_otimizado_calm, axis=0)
    erro_maximo_otimizado_calm = np.max(erros_matriz_otimizado_calm, axis=0)
    erro_minimo_otimizado_calm = np.min(erros_matriz_otimizado_calm, axis=0)

    # orientação global
    erros_matriz_ori_otimizado_calm = np.array(erros_historico_ori_otimizado_calm)
    erro_medio_ori_calm = np.mean(erros_matriz_ori_otimizado_calm, axis=0)
    erro_max_ori_calm = np.max(erros_matriz_ori_otimizado_calm, axis=0)
    erro_min_ori_calm = np.min(erros_matriz_ori_otimizado_calm, axis=0)


    # SIMULAÇÕES HIL
    ### STORM
    print("\n--- STORM ---\n")
    enviar_ganhos_stm32(best_kp_storm, best_ki_storm, best_kd_storm, nome_cenario="STORM")

    erros_historico_otimizado_storm = []
    erros_historico_ori_otimizado_storm = []
    df_first_storm = None

    for i, alvo in enumerate(ten_targets):
        print(f"Executando alvo STORM HIL {i+1}/{len(ten_targets)}...")
        df_target, _ = run_experiment_storm(best_kp_storm, best_ki_storm, best_kd_storm, target_pos=alvo)
        if i == 0:
            df_first_storm = df_target
        erros_historico_otimizado_storm.append(df_target['error_pos_norm'].values)
        erros_historico_ori_otimizado_storm.append(df_target['error_ori_norm'].values)

    time_vector_storm = df_first_storm['time']

    erros_matriz_otimizado_storm = np.array(erros_historico_otimizado_storm)
    erro_medio_otimizado_storm = np.mean(erros_matriz_otimizado_storm, axis=0)
    erro_maximo_otimizado_storm = np.max(erros_matriz_otimizado_storm, axis=0)
    erro_minimo_otimizado_storm = np.min(erros_matriz_otimizado_storm, axis=0)

    # orientação global
    erros_matriz_ori_otimizado_storm = np.array(erros_historico_ori_otimizado_storm)
    erro_medio_ori_storm = np.mean(erros_matriz_ori_otimizado_storm, axis=0)
    erro_max_ori_storm = np.max(erros_matriz_ori_otimizado_storm, axis=0)
    erro_min_ori_storm = np.min(erros_matriz_ori_otimizado_storm, axis=0)


    # PLOTAGEM DOS GRÁFICOS
    # 1: Erro de Posição 3D (Calm vs Storm HIL)
    plt.figure(figsize=(8, 5))

    plt.plot(time_vector_calm, erro_medio_otimizado_calm, label='Mean Error (Calm HIL)', color='blue', linewidth=2.0)
    plt.fill_between(time_vector_calm, erro_minimo_otimizado_calm, erro_maximo_otimizado_calm, color='blue', alpha=0.15, label='Bounds (Calm HIL)')
    
    plt.plot(time_vector_storm, erro_medio_otimizado_storm, label='Mean Error (Storm HIL)', color='red', linewidth=2.0, linestyle='--')
    plt.fill_between(time_vector_storm, erro_minimo_otimizado_storm, erro_maximo_otimizado_storm, color='red', alpha=0.15, label='Bounds (Storm HIL)')
    
    plt.title("3D Position Error Comparison (Calm vs Storm HIL)", fontsize=13, pad=10)
    
    plt.xlabel("Time [s]")
    plt.ylabel("3D Position Error [m]")
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_tracking_error_position_hil_calm_storm")

    # 2: Erro de Orientação 3D (Calm vs Storm HIL)
    plt.figure(figsize=(8, 5))

    plt.plot(time_vector_calm, erro_medio_ori_calm, label='Mean Error (Calm HIL)', color='blue', linewidth=2.0)
    plt.fill_between(time_vector_calm, erro_min_ori_calm, erro_max_ori_calm, color='blue', alpha=0.15, label='Bounds (Calm HIL)')
    
    plt.plot(time_vector_storm, erro_medio_ori_storm, label='Mean Error (Storm HIL)', color='red', linewidth=2.0, linestyle='--')
    plt.fill_between(time_vector_storm, erro_min_ori_storm, erro_max_ori_storm, color='red', alpha=0.15, label='Bounds (Storm HIL)')
    
    plt.title("3D Orientation Error Comparison (Calm vs Storm HIL)", fontsize=13, pad=10)
    
    plt.xlabel("Time [s]")
    plt.ylabel("3D Orientation Error [deg]")
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_tracking_error_orientation_hil_calm_storm")

    # 3: Esforço de Controle
    plt.figure(figsize=(8, 5))

    plt.plot(df_first_calm['time'], df_first_calm['esforco_total'], label='Control Effort (Calm HIL)', color='blue', linewidth=2.0)
    plt.plot(df_first_storm['time'], df_first_storm['esforco_total'], label='Control Effort (Storm HIL)', color='red', linewidth=2.0)
    
    plt.title("Total Control Effort Comparison (Calm vs Storm HIL)", fontsize=13, pad=10)
    
    plt.xlabel("Time [s]")
    plt.ylabel("Total Control Effort [N / Nm]")
    plt.legend(frameon=True)
    plt.tight_layout()
    save_plot("fig_control_effort_hil_calm_storm")

    # Métricas
    rmse_medio_otimizado_calm = np.sqrt(np.mean(erro_medio_otimizado_calm**2))
    rmse_medio_otimizado_storm = np.sqrt(np.mean(erro_medio_otimizado_storm**2))

    print("\n" + "="*40)
    print("--- RESUMO DAS MÉTRICAS HIL ---")
    print(f"RESUMO CALM HIL: RMSE Posição 3D Média = {rmse_medio_otimizado_calm:.4f} m")
    print(f"RESUMO STORM HIL: RMSE Posição 3D Média = {rmse_medio_otimizado_storm:.4f} m")
    print("="*40 + "\n")

    plt.show()

    if ser and ser.is_open:
        ser.close()
        print("[OK] Porta serial fechada.")