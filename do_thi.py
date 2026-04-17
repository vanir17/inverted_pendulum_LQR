import numpy as np
import mujoco
import mujoco.viewer
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import time

# 1. Định nghĩa mô hình XML (Con lắc ngược trên xe trượt - Cartpole)
xml_content = """
<mujoco model="inverted pendulum">
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <worldbody>
        <light pos="0 0 5"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1"/>
        <body name="cart" pos="0 0 0.1">
            <joint name="slider" type="slide" axis="1 0 0"/>
            <geom name="cart_geom" type="box" size="0.2 0.1 0.05" rgba="0 0 1 1"/>
            <body name="pole" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="1 0 0 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="slide_motor" joint="slider" gear="1" ctrlrange="-100 100"/>
    </actuator>
</mujoco>
"""

# 2. Khởi tạo Model và Data
model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

# 3. Hàm tính toán ma trận tuyến tính hóa A, B
def get_linear_model(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.0  # Vị trí cân bằng (thẳng đứng)
    mujoco.mj_forward(model, data)

    A = np.zeros((model.nv * 2, model.nv * 2))
    B = np.zeros((model.nv * 2, model.nu))

    # Tính toán sai phân hữu hạn (Finite Difference) để lấy A, B rời rạc
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B

# 4. Hàm tính toán độ lợi LQR
def calculate_lqr_gain(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

# Thiết lập thông số LQR
A, B = get_linear_model(model, data)
Q = np.diag([50, 500, 1, 200]) # Ưu tiên giữ góc (500) và vị trí xe (50)
R = np.array([[3]])           # Trọng số cho năng lượng điều khiển
K = calculate_lqr_gain(A, B, Q, R)

# 5. Cấu hình mô phỏng
mujoco.mj_resetData(model, data)
data.qpos[1] = 0.2  # Độ lệch ban đầu: 0.2 rad (~11.5 độ)
x_target = np.array([0.0, 0.0, 0.0, 0.0])

# Khởi tạo lưu trữ dữ liệu để vẽ đồ thị
history = {
    'time': [],
    'cart_pos': [],
    'pole_angle': [],
    'cart_vel': [],
    'pole_vel': []
}

print("Đang khởi động mô phỏng và lưu dữ liệu...")

# 6. Vòng lặp mô phỏng
sim_duration = 15.0 # Mô phỏng trong 5 giây
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < sim_duration:
        step_start = time.time()

        # Trạng thái hiện tại: [vị trí xe, góc gậy, vận tốc xe, vận tốc góc]
        x = np.concatenate([data.qpos, data.qvel])
        
        # Luật điều khiển LQR: u = -K * (x - x_target)
        u = - K @ (x - x_target)
        data.ctrl[0] = np.clip(u[0], -100, 100)

        # Lưu dữ liệu
        history['time'].append(data.time)
        history['cart_pos'].append(x[0])
        history['pole_angle'].append(x[1])
        history['cart_vel'].append(x[2])
        history['pole_vel'].append(x[3])

        # Bước mô phỏng
        mujoco.mj_step(model, data)
        viewer.sync()

        # Duy trì tốc độ thời gian thực
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Mô phỏng hoàn tất. Đang vẽ đồ thị...")

# 7. Vẽ đồ thị đáp ứng
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Đồ thị đáp ứng hệ thống Inverted Pendulum (LQR Control)', fontsize=16)

# Vị trí xe
axs[0, 0].plot(history['time'], history['cart_pos'], 'b', linewidth=1.5)
axs[0, 0].set_title('Cart Position (m)')
axs[0, 0].grid(True, linestyle='--')
axs[0, 0].axhline(0, color='black', lw=1)

# Góc con lắc
axs[0, 1].plot(history['time'], history['pole_angle'], 'r', linewidth=1.5)
axs[0, 1].set_title('Pole Angle (rad)')
axs[0, 1].grid(True, linestyle='--')
axs[0, 1].axhline(0, color='black', lw=1)

# Vận tốc xe
axs[1, 0].plot(history['time'], history['cart_vel'], 'g', linewidth=1.5)
axs[1, 0].set_title('Cart Velocity (m/s)')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].grid(True, linestyle='--')

# Vận tốc góc
axs[1, 1].plot(history['time'], history['pole_vel'], 'm', linewidth=1.5)
axs[1, 1].set_title('Pole Angular Velocity (rad/s)')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].grid(True, linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()