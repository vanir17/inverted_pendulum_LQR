import numpy as np
import mujoco
import mujoco.viewer
from scipy.linalg import solve_discrete_are
import time

model = mujoco.MjModel.from_xml_path("env.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model,data)

Q_goal = np.diag([200, 500, 10, 100]) #x, theta, x_dot, theta_dot
R_goal = np.array([1])

# 2. SPSA constants - SPALL CONSTANTS
alpha = 0.602 #Decay rate
gamma = 0.101 
a_gain = 0.6    # Learning rate
c_gain = 0.5   # Pertubation value
A_bar = 5       # Stable delay 


def cost_evaluation(theta):
    Q_design = np.diag(Q_goal.diagonal() * np.maximum(theta, 0.01))

    try:
        # Solve Ricatti Equation to find gain K
        P = solve_discrete_are(A_nominal, B_nominal, Q_design, R_goal)
        K = np.linalg.inv(R_goal + B_nominal.T @ P @ B_nominal) @ (B_nominal.T @ P @ A_nominal)
    except:
        # Nếu bộ theta này làm hệ thống không giải được LQR -> Phạt rất nặng
        return 1e8

    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.1
    total_cost = 0
    duration_steps = 3000
    
    for k in range(0, duration_steps):
        x = np.concatenate([data.qpos, data.qvel])

        u = -K @ x
        data.ctrl[0] = np.clip(u[0], -100, 100)

        mujoco.mj_step(model, data)
        
        step_cost = x.T @ Q_goal @ x + u[0] * R_goal * u[0]
        total_cost = total_cost + step_cost
        # Trong hàm cost_evaluation
        if abs(data.qpos[1]) > 0.8 or abs(data.qpos[0]) > 2.0: # Phạt nếu đổ HOẶC chạy quá xa
            total_cost += 10000 * (duration_steps - k)
            break
    return total_cost / duration_steps

def get_linear_model(model, data):
    
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)

    # create matrix 0 for mujoco fill
    A = np.zeros((model.nv * 2, model.nv * 2))
    B = np.zeros((model.nv * 2, model.nu))

    #calculate finite differential values
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B

A, B = get_linear_model(model, data)

#Discrepancies from true values: 10% 
A_nominal = A
B_nominal = B * 0.9

theta = np.ones(4)
theta = np.maximum(theta, 1e-3)
Q_bar = Q_goal * theta

i_max = 50
ng = 2

for i in range (0, i_max):

    a_i = a_gain / (i + 1 + A_bar)**alpha
    c_i = c_gain / (i + 1)**gamma

    gradients = []

    for j in range(0, ng):

        delta = np.random.choice([-1,1], size = 4)

        theta_plus = np.maximum(theta + c_i * delta, 0.01)
        j_plus = cost_evaluation(theta_plus)

        theta_minus = np.maximum(theta - c_i * delta, 0.01)
        j_minus = cost_evaluation(theta_minus)

        g_ij = (j_plus - j_minus) / (2 * c_i * delta)
        gradients.append(g_ij)

    g_i = np.mean(gradients, axis = 0)
    theta = theta - a_i * g_i
    theta = np.maximum(theta, 0.01)
    print(f"Loop {i}: J = {j_plus}, Theta = {theta}")


Q_final = np.diag(Q_goal.diagonal() * theta)
P = solve_discrete_are(A_nominal, B_nominal, Q_final, R_goal)
K_final = np.linalg.inv(R_goal + B_nominal.T @ P @ B_nominal) @ (B_nominal.T @ P @ A_nominal)

x_target = np.array([0.0,0.0,0.0,0.0])

print("Matrix K:\n", K_final)#caculate matrix gain K - LQR
print("Matrix A:\n", A_nominal)#caculate matrix gain K - LQR
print("Matrix B:\n", B_nominal)#caculate matrix gain K - LQR

data.qpos[1] = 1

while viewer.is_running():
    step_start = time.time()

    x = np.concatenate([data.qpos, data.qvel])
    u = - K_final @ (x - x_target)
    data.ctrl[0] = np.clip(u[0], -100,100)
    # print(f"LQR Gain K: {K}")
    mujoco.mj_step(model, data)
    viewer.sync()

    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)









