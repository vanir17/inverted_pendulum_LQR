import numpy as np
import mujoco
import mujoco.viewer
from scipy.linalg import solve_discrete_are
import time

# --- CẤU HÌNH ---
model = mujoco.MjModel.from_xml_path("env.xml")
data = mujoco.MjData(model)

# 1. Ma trận Q, R mục tiêu (Dùng để tính điểm - Performance Weights)
Q_goal = np.diag([50, 500, 1, 20])
R_goal = np.array([[1]])

# 2. Tham số SPSA (Spall's constants)
alpha = 0.602
gamma = 0.101
a_gain = 0.4    # Tốc độ học
c_gain = 0.05   # Độ "rung" thử nghiệm
A_bar = 5       # Độ trễ ổn định

def get_linear_model(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)
    A = np.zeros((model.nv * 2, model.nv * 2))
    B = np.zeros((model.nv * 2, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B

# Giả sử mô hình toán học (A, B) bị sai lệch 20% so với thực tế (Model Mismatch)
A_nom, B_nom = get_linear_model(model, data)
A_nom *= 1.2 
B_nom *= 0.8

def cost_evaluation(theta):
    """ Hàm chạy thử nghiệm robot và trả về điểm số J (Dòng 18-21) """
    # Tạo ma trận thiết kế Q_bar dựa trên theta
    Q_design = np.diag(Q_goal.diagonal() * np.maximum(theta, 0.01))
    
    # Thiết kế bộ LQR dựa trên mô hình SAI và ma trận Q_design
    try:
        P = solve_discrete_are(A_nom, B_nom, Q_design, R_goal)
        K = np.linalg.inv(R_goal + B_nom.T @ P @ B_nom) @ (B_nom.T @ P @ A_nom)
    except: return 1e6 # Nếu không giải được, phạt nặng

    # Chạy mô phỏng 2 giây để đo hiệu suất
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.1 # Cho lệch 0.1 rad
    cost = 0
    for _ in range(200): # 200 bước mô phỏng
        x = np.concatenate([data.qpos, data.qvel])
        u = -K @ x
        data.ctrl[0] = np.clip(u[0], -100, 100)
        mujoco.mj_step(model, data)
        
        # Tính chi phí dựa trên Q_goal (Mục tiêu thực sự)
        cost += (x.T @ Q_goal @ x + u.T @ R_goal @ u)
    return cost / 200

# --- VÒNG LẶP CHÍNH SPSA (Dòng 2 - 15) ---
theta = np.ones(4) # Khởi tạo theta_0 = [1, 1, 1, 1]
i_max = 30

print("Bắt đầu Self-Tuning...")
for i in range(i_max):
    # Bước 3-4: Cập nhật bước nhảy
    a_i = a_gain / (i + 1 + A_bar)**alpha
    c_i = c_gain / (i + 1)**gamma
    
    # Bước 6: Tạo nhiễu Bernoulli ngẫu nhiên (+1 hoặc -1)
    delta = np.where(np.random.rand(4) > 0.5, 1, -1)
    
    # Bước 7-10: Thử nghiệm J+ và J-
    j_plus = cost_evaluation(theta + c_i * delta)
    j_minus = cost_evaluation(theta - c_i * delta)
    
    # Bước 11-13: Ước lượng Gradient (Độ dốc)
    g_i = (j_plus - j_minus) / (2 * c_i * delta)
    
    # Bước 14: Cập nhật theta
    theta = theta - a_i * g_i
    theta = np.maximum(theta, 0.01) # Đảm bảo theta không âm
    
    print(f"Vòng {i}: J = {j_plus:.4f}, Theta = {theta}")

print(f"\nTheta tối ưu tìm được: {theta}")