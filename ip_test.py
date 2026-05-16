import mujoco
import mujoco.viewer
import time

# 1. Định nghĩa chuỗi XML (đã sửa một vài lỗi cú pháp nhỏ trong file của bạn)
xml_content = """
<mujoco model="inverted_pendulum">
    <compiler angle="radian" autolimits="true" inertiafromgeom="true"/>

    <option timestep="0.005"
            gravity="0 0 -9.81"
            integrator="RK4"/>
    <visual>
        <global azimuth="130" elevation="-30"/>
    </visual>

    <asset> 
        <material name="rail_mat" rgba="0.2 0.2 0.2 1"/>
        <material name="cart_mat" rgba="0.0 0.4 0.8 1"/>
        <material name="pole_mat" rgba="0.8 0.1 0.1 1"/>
        <material name="floor_mat" rgba="0.9 0.9 0.9 1"/>
    </asset>

    <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" size="1.0 0.5 0.02" material="floor_mat"/>

        <!-- MGN12C Rail -->
        <geom name="mgn12_rail"
                type="box"
                pos="0 0 0.03"
                size="0.25 0.006 0.006" 
                material="rail_mat"/>

        <body name="cart" pos="0 0 0.045">
            <joint name="cart_slide"
                    type="slide"
                    axis="1 0 0"
                    limited="true"
                    range="-0.225 0.225"
                    damping="0.15"
                    frictionloss="0.4"
                    armature="0.06"/>
            <geom name="cart_body"
                    type="box"
                    size="0.035 0.03 0.015"
                    mass="0.3"
                    material="cart_mat"/>
            
            <body name="pole" pos="0 0 0.015">
                <joint name="pole_hinge"
                    type="hinge"
                    axis="0 1 0"
                    limited="false"
                    damping="0.001"
                    frictionloss="0.005"
                    armature="0.001"/>
                <geom name="pole_rod"
                    type="capsule"
                    fromto="0 0 0 0 0 0.30"
                    size="0.005"
                    mass="0.05"
                    material="pole_mat"/>
                <geom name="pole_tip"
                    type="sphere"
                    pos="0 0 0.30"
                    size="0.012"
                    mass="0.02"
                    rgba="0.9 0.6 0.0 1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="nema17_actuator"
               joint="cart_slide"
               gear="1"
               ctrllimited="true"
               ctrlrange="-40 40"/>
    </actuator>
</mujoco>
"""

# 2. Khởi tạo mô hình MuJoCo từ chuỗi XML
model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

# Thay đổi tư thế ban đầu: cho con lắc lệch đi 0.1 radian để nó tự rơi xuống
data.qpos[1] = 0.1 

print("Đang khởi động cửa sổ mô phỏng MuJoCo...")
print("Mẹo: Bạn có thể nhấn giữ CHUỘT PHẢI để kéo thả con lắc trong không gian mô phỏng.")

# 3. Mở cửa sổ Viewer và chạy vòng lặp vật lý
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Giữ cửa sổ luôn mở cho đến khi người dùng tắt
    while viewer.is_running():
        step_start = time.time()

        # Thực hiện 1 bước tính toán vật lý (0.005 giây như cấu hình timestep)
        mujoco.mj_step(model, data)

        # Cập nhật hình ảnh lên màn hình
        viewer.sync()

        # Đồng bộ thời gian thực để mô phỏng không chạy quá nhanh so với thực tế
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)