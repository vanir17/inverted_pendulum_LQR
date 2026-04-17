import mujoco.viewer
import numpy as np
import scipy.linalg
import time

# ---------------------------------------------------------
# 1. Cấu hình mô hình MuJoCo (XML)
# ---------------------------------------------------------
xml_content = """
<mujoco model="cartpole">
    <default>
        <joint damping="0.05"/>
        <geom friction="1 0.1 0.1" rgba="0.7 0.7 0.7 1"/>
    </default>
    <worldbody>
        <light directional="true" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 .05"/>
        <body name="cart" pos="0 0 0.2">
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
            <joint name="slider" type="slide" pos="0 0 0" axis="1 0 0" damping="0.1"/>
            <geom name="cart_geom" type="box" size="0.1 0.05 0.05" rgba="0 0.5 0.8 1"/>
            <body name="pole" pos="0 0 0">
                <inertial pos="0 0 0.3" mass="0.2" diaginertia="0.006 0.006 0.0001"/>
                <joint name="hinge" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.001"/>
                <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.01" rgba="0.8 0.2 0.2 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="force" joint="slider" gear="1" ctrlrange="-100 100" ctrllimited="true"/>
    </actuator>
</mujoco>
"""

# ---------------------------------------------------------
# 2. Các hàm hỗ trợ Điều khiển
# ---------------------------------------------------------
def solve_lqr(A, B, Q, R):
    """Giải phương trình Riccati rời rạc để tìm Gain F"""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    F = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return F

def get_nominal_AB(model, data):
    """Tuyến tính hóa MuJoCo để lấy ma trận A, B"""
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0 # Đứng thẳng
    mujoco.mj_forward(model, data)
    
    A = np.zeros((model.nv * 2, model.nv * 2))
    B = np.zeros((model.nv * 2, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, False, A, B, None, None)
    return A, B

# ---------------------------------------------------------
# 3. Thuật toán Self-Tuning LQR
# ---------------------------------------------------------
class SelfTuningLQR:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.A, self.B = get_nominal_AB(model, data)
        
        # Q, R Tiêu chuẩn (Performance Weights)
        self.Q_base = np.diag([1.0, 1.0, 50.0, 1.0]) # Ưu tiên phạt góc (vị trí thứ 3)
        self.R_base = np.array([[0.1]])
        
        # SPSA Hyperparameters
        self.alpha = 0.602
        self.gamma = 0.101
        self.a_gain = 0.5
        self.c_gain = 0.1
        self.A_cap = 5
        
        # Khởi tạo theta (Dòng 1 Algorithm 1)
        self.theta = np.ones(4) # [theta_x, theta_v, theta_angle, theta_omega]

    def cost_evaluation(self, theta_val):
        """Hàm J_hat: Chạy simulation và tính chi phí (Dòng 18-21)"""
        # Đảm bảo theta dương (Projection)
        theta_val = np.maximum(theta_val, 0.01)
        
        # Tạo Q_bar, R_bar
        Q_bar = np.diag(self.Q_base.diagonal() * theta_val)
        R_bar = self.R_base
        
        try:
            F = solve_lqr(self.A, self.B, Q_bar, R_bar)
        except:
            return 1e6 # Trả về chi phí rất cao nếu không giải được LQR

        # Thực hiện experiment (Dòng 20)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = 0.1 # Khởi tạo lệch 0.1 rad để thử thách
        
        total_cost = 0
        steps = 300 # Chạy 3 giây (300 steps với dt=0.01)
        
        for _ in range(steps):
            x = np.concatenate([self.data.qpos, self.data.qvel])
            u = -F @ x
            self.data.ctrl[0] = u[0]
            
            mujoco.mj_step(self.model, self.data)
            
            # Tính chi phí dựa trên Q, R tiêu chuẩn
            step_cost = x.T @ self.Q_base @ x + u.T @ self.R_base @ u
            total_cost += step_cost
            
            if abs(self.data.qpos[1]) > 1.0: # Ngã rồi!
                total_cost += 1000 # Phạt nặng
                break
                
        return total_cost / steps

    def train(self, i_max=20):
        print(f"Bắt đầu Self-tuning SPSA...")
        for i in range(i_max):
            a_i = self.a_gain / (i + 1 + self.A_cap)**self.alpha
            c_i = self.c_gain / (i + 1)**self.gamma
            
            # Dòng 6: Bernoulli distribution
            delta = np.where(np.random.rand(4) > 0.5, 1, -1)
            
            # Dòng 7-10: Đánh giá J+ và J-
            theta_plus = self.theta + c_i * delta
            j_plus = self.cost_evaluation(theta_plus)
            
            theta_minus = self.theta - c_i * delta
            j_minus = self.cost_evaluation(theta_minus)
            
            # Dòng 11: Ước lượng Gradient
            g_i = (j_plus - j_minus) / (2 * c_i * delta)
            
            # Dòng 14: Cập nhật theta
            self.theta = self.theta - a_i * g_i
            self.theta = np.maximum(self.theta, 0.01) # Chặn dưới
            
            print(f"Vòng {i+1}/{i_max} | J: {j_plus:.2f} | Theta: {self.theta}")

# ---------------------------------------------------------
# 4. Thực thi chính
# ---------------------------------------------------------
if __name__ == "__main__":
    # Khởi tạo mô hình
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    # 1. Chạy quá trình training SPSA để tìm Theta tối ưu
    tuner = SelfTuningLQR(model, data)
    tuner.train(i_max=30)
    
    print("\n--- BẮT ĐẦU MÔ PHỎNG TRỰC QUAN ---")
    
    # 2. Tính toán Gain F cuối cùng từ Theta đã học được
    Q_final = np.diag(tuner.Q_base.diagonal() * tuner.theta)
    R_final = tuner.R_base
    F_final = solve_lqr(tuner.A, tuner.B, Q_final, R_final)
    
    # 3. Mở Viewer và chạy mô phỏng thời gian thực
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset trạng thái về vị trí ban đầu (lệch một chút để thử thách bộ điều khiển)
        mujoco.mj_resetData(model, data)
        data.qpos[1] = 0.2  # Lệch 0.2 rad (~11 độ)
        
        print("Đang chạy mô phỏng... Nhấn Ctrl+C trong terminal hoặc đóng cửa sổ để dừng.")
        
        while viewer.is_running():
            x = np.concatenate([data.qpos, data.qvel])
    
            # 1. Tính toán lực LQR bình thường
            u = -F_final @ x
            
            # 2. Safety Layer: Kiểm tra giới hạn biên (Virtual Bumper)
            cart_pos = data.qpos[0]
            limit = 0.95 # Giới hạn an toàn (hơi nhỏ hơn 1.0m của XML)
            
            if cart_pos > limit:
                # Nếu xe quá sát biên phải, ép lực không được dương (không cho đẩy thêm sang phải)
                u[0] = min(u[0], 0) - 10 * (cart_pos - limit) # Thêm lực đẩy ngược nhẹ
            elif cart_pos < -limit:
                # Nếu xe quá sát biên trái, ép lực không được âm
                u[0] = max(u[0], 0) + 10 * (-limit - cart_pos)

            # 3. Áp dụng lực đã qua xử lý
            data.ctrl[0] = u[0]
            
            mujoco.mj_step(model, data)
            viewer.sync()
            