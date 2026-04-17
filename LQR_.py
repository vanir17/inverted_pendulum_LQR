import numpy as np
import mujoco
import mujoco.viewer
from scipy.linalg import solve_discrete_are
import time 

model = mujoco.MjModel.from_xml_path("env.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model,data)

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

print("Matrix B:\n", B)#caculate matrix gain K - LQR
print("Matrix a:\n", A)#caculate matrix gain K - LQR

def calculate_lqr_gain(A, B, Q, R):

    P = solve_discrete_are(A, B, Q, R) #Calculate discrete Algebraic Ricatti Equation (ARE)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


Q = np.diag([50, 500, 1, 200])
R = np.array([[3]])
K = calculate_lqr_gain(A, B, Q, R)
data.qpos[1] = 0.2
x_target = np.array([0.0,0.0,0.0,0.0])
print("Simulating...\n\n")
print(f"LQR Gain K: {K}")

while viewer.is_running():
    step_start = time.time()

    x = np.concatenate([data.qpos, data.qvel])
    u = - K @ (x - x_target)
    data.ctrl[0] = np.clip(u[0], -100,100)
    # print(f"LQR Gain K: {K}")
    mujoco.mj_step(model, data)
    viewer.sync()

    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)



    