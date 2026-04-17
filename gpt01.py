import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.linalg import solve_discrete_are

XML_PATH = "inverted_pendulum.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

def get_linear_model(model, data):
    mujoco.mj_resetData(model, data)

    # tuyến tính hóa quanh trạng thái upright
    data.qpos[:] = np.array([0.0, 0.0])
    data.qvel[:] = np.array([0.0, 0.0])
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    nx = 2 * model.nv + model.na
    A = np.zeros((nx, nx))
    B = np.zeros((nx, model.nu))

    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B

def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

A, B = get_linear_model(model, data)

# state theo thứ tự MuJoCo ở đây là:
# [x, theta, x_dot, theta_dot]
Q = np.diag([10, 5000, 0, 0])
R = np.array([[1.0]])

K = dlqr(A, B, Q, R)

print("A =\n", A)
print("B =\n", B)
print("K =\n", K)

# reset trạng thái đầu
mujoco.mj_resetData(model, data)
data.qpos[:] = np.array([0.0, 0.08])   # lệch góc nhỏ
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()

        x = np.array([
            data.qpos[0],
            data.qpos[1],
            data.qvel[0],
            data.qvel[1]
        ])

        u = -K @ x
        data.ctrl[0] = np.clip(float(u[0]), -1000, 1000)

        mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)