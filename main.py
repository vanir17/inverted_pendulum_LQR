import mujoco
import mujoco_viewer
import numpy as np
import scipy.linalg
import time

def get_ab_matrices(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[1] = np.pi
    mujoco.mj_forward(model, data)
    
    nv = model.nv
    nu = model.nu
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    
    mujoco.mjd_transitionFD(model, data, 1e-6, False, A, B, None, None)
    return A, B

def calculate_lqr_gain(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("cartpole.xml")
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    A, B = get_ab_matrices(model, data)
    
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)

    Q = np.diag([1.0, 100.0, 1.0, 10.0])
    R = np.array([[0.1]])
    
    K = calculate_lqr_gain(A, B, Q, R)
    
    print("LQR Gain K:")
    print(K)

    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.15 
    
    print("Starting simulation loop...")

    while viewer.is_alive:
        x = np.concatenate((data.qpos, data.qvel))
        
        theta_err = ((data.qpos[1] - np.pi + np.pi) % (2*np.pi)) - np.pi
        x = np.array([data.qpos[0], theta_err, data.qvel[0], data.qvel[1]])
        u = -K @ x
        
        data.ctrl[0] = u[0]
        
        mujoco.mj_step(model, data)
        viewer.render()
        
        time.sleep(0.01)