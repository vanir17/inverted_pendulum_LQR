import mujoco
import mujoco.viewer
xml_content = """
<mujoco model="inverted_pendulum">
    <default>
        <joint damping="0.05" armature="0.01"/>
        <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0.7 1"/>
    </default>
    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 .05" rgba="0.8 0.9 0.8 1"/>
        <geom name="rail" type="capsule" pos="0 0 0.2" quat="0.707 0 0.707 0" size="0.02 2" 
            rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
        <body name="cart" pos="0 0 0.2">
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.01 0.01"/>
            <joint name="slide_joint" type="slide" pos="0 0 0" axis="1 0 0" limited="true" range="-3 3" damping="0.1"/>
            <geom name="cart_geom" type="box" size="0.1 0.05 0.05" rgba="0 0.5 0.8 1"/>
            <body name="pole" pos="0 0 0">
                <inertial pos="0 0 0.3" mass="0.2" diaginertia="0.006 0.006 0.0001"/>
                <joint name="hinge" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.001"/>
                <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.015" rgba="0.8 0.2 0.2 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="force_actuator" joint="slide_joint" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
    </actuator>
</mujoco>
"""
print("Loading model...")
model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)
print("Launching passive...")
viewer = mujoco.viewer.launch_passive(model, data)
print("Done")
viewer.close()
