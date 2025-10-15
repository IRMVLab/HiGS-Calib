import open3d as o3d
import numpy as np
import threading
import time
import copy  # For deep copying pose data
from sensors.sensor_trajectories import SensorTrajectories
import pytorch3d
from matplotlib import pyplot as plt


# --- Helper Functions (unchanged) ---
def create_transformation_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def get_rotation_matrix_from_xyz_angles(rx, ry, rz, in_degrees=False):
    if in_degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def create_visualized_pose_geometries(pose_matrix, frame_size=0.5, sphere_radius=0.05, sphere_color=None):
    geometries = []
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])
    coordinate_frame.transform(pose_matrix)
    geometries.append(coordinate_frame)

    if sphere_color is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.paint_uniform_color(sphere_color)
        sphere.transform(pose_matrix)
        geometries.append(sphere)
    return geometries


# --- Visualizer Thread Class ---
class PoseVisualizerThread(threading.Thread):
    def __init__(self, window_name="Pose Visualization"):
        super().__init__(daemon=True)
        self.window_name = window_name
        self.vis = o3d.visualization.Visualizer()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        self._static_pose_matrices_for_view_calc = []  # Store matrices for initial view calculation
        self._static_o3d_geometries = []
        self._camera_current_poses_matrices = []
        self._camera_current_o3d_geometries = []

        self._needs_add_static = True
        self._needs_update_current_cam = True
        self._initial_view_set = False

        self._is_initialized = False

    def _initialize_visualizer(self):
        self.vis.create_window(window_name=self.window_name)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.9, 0.9, 0.9])
        # Add a world frame by default
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # Don't reset bounding box yet, wait for other geometries
        self.vis.add_geometry(world_frame, reset_bounding_box=False)
        self._is_initialized = True

    def _set_initial_camera_view(self):
        if not self._static_pose_matrices_for_view_calc:
            self.vis.reset_view_point(True)  # Default if no poses
            return

        all_origins = []
        for pose_matrix_list in self._static_pose_matrices_for_view_calc:
            for pose_matrix in pose_matrix_list:
                all_origins.append(pose_matrix[:3, 3])

        if not all_origins:  # Should not happen if _static_pose_matrices_for_view_calc is populated
            self.vis.reset_view_point(True)
            return

        all_origins_np = np.array(all_origins)
        scene_center = np.mean(all_origins_np, axis=0)

        # Calculate scene extents to determine camera distance
        min_coords = np.min(all_origins_np, axis=0)
        max_coords = np.max(all_origins_np, axis=0)
        scene_diameter = np.linalg.norm(max_coords - min_coords)
        if scene_diameter < 1e-3:  # Avoid division by zero or tiny scenes
            scene_diameter = 5.0  # Default diameter

        view_ctrl = self.vis.get_view_control()

        # Look at the center of the scene
        lookat = scene_center

        # Position camera: From a point along +Y axis relative to scene_center, and slightly elevated
        # The distance is proportional to the scene diameter
        eye_offset_y = scene_diameter * 1.5  # Distance along Y
        eye_offset_z = scene_diameter * 0.5  # Elevation

        eye = scene_center + np.array([0, eye_offset_y, eye_offset_z])

        # Camera's "front" vector is from eye to lookat
        front_vec = lookat - eye
        # Ensure it's not zero length if eye and lookat coincide
        if np.linalg.norm(front_vec) < 1e-6:
            front_vec = np.array([0, -1, -0.1])  # Default if eye is at lookat

        up_vec = np.array([0, 0, 1])  # World Z is up for the camera view

        # Set view using extrinsic parameters (eye, lookat, up)
        # Note: Open3D's set_lookat, set_front, set_up is more direct for this
        view_ctrl.set_lookat(lookat)
        view_ctrl.set_front(front_vec / np.linalg.norm(front_vec))  # Normalized front vector
        view_ctrl.set_up(up_vec)
        view_ctrl.set_zoom(0.7)  # Adjust zoom as needed; 0.7 often works well

        print(f"Set initial view: lookat={lookat}, eye={eye}, up={up_vec}")

    def run(self):
        self._initialize_visualizer()

        while not self._stop_event.is_set():
            update_occurred = False
            initial_static_added_this_cycle = False

            with self.lock:
                if self._needs_add_static:
                    for geo_group in self._static_o3d_geometries:  # This should be populated by add_static_pose_group
                        for geo_list in geo_group:  # geo_group is a list of lists of geometries
                            for geo in geo_list:
                                self.vis.add_geometry(geo, reset_bounding_box=False)
                    self._needs_add_static = False
                    update_occurred = True
                    initial_static_added_this_cycle = True

                if self._needs_update_current_cam:
                    # Remove old current camera pose geometries
                    for geo_list in self._camera_current_o3d_geometries:
                        for geo in geo_list:
                            self.vis.remove_geometry(geo, reset_bounding_box=False)
                    self._camera_current_o3d_geometries.clear()

                    # Add new current camera pose geometries
                    for pose_matrix in self._camera_current_poses_matrices:
                        new_geos = create_visualized_pose_geometries(
                            pose_matrix,
                            frame_size=0.6, sphere_radius=0.06, sphere_color=[1, 0, 0]
                        )
                        self._camera_current_o3d_geometries.append(new_geos)
                        for geo in new_geos:
                            self.vis.add_geometry(geo, reset_bounding_box=False)
                    self._needs_update_current_cam = False
                    update_occurred = True

            if initial_static_added_this_cycle and not self._initial_view_set:
                self._set_initial_camera_view()
                self._initial_view_set = True
            elif update_occurred and not self._initial_view_set:  # If current cam updated before static ones
                self.vis.reset_view_point(True)  # Fallback if only dynamic poses are there initially

            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

        self.vis.destroy_window()
        print("Visualizer thread stopped.")

    def add_static_pose_group(self, pose_matrices_list, frame_size, sphere_radius, sphere_color):
        with self.lock:
            # Store raw matrices for view calculation if it's the first static group
            if not self._static_pose_matrices_for_view_calc:  # Store all groups for more robust center
                self._static_pose_matrices_for_view_calc.append(list(pose_matrices_list))  # list() to copy
            else:  # Append to existing lists
                self._static_pose_matrices_for_view_calc[0].extend(list(pose_matrices_list))

            geo_group_for_vis = []  # This will be a list of [frame, sphere] lists
            for pose_matrix in pose_matrices_list:
                geos = create_visualized_pose_geometries(pose_matrix, frame_size, sphere_radius, sphere_color)
                geo_group_for_vis.append(geos)
            self._static_o3d_geometries.append(geo_group_for_vis)  # Add the group to the list of static groups
            self._needs_add_static = True
            self._initial_view_set = False  # Re-trigger view setup if new static poses are added

    def update_camera_current_poses(self, new_pose_matrices):
        with self.lock:
            self._camera_current_poses_matrices = [copy.deepcopy(p) for p in new_pose_matrices]
            self._needs_update_current_cam = True

    def stop(self):
        self._stop_event.set()


def visualize(scene):
    # Visualize
    pose_viz_thread = PoseVisualizerThread()
    pose_viz_thread.start()

    # --- 2. Prepare and add static pose data ---
    # == LiDAR 位姿真值 (World to LiDAR) ==
    lidar_poses_gt_matrices = []
    for i in range(scene.sensor_trajectory.get_length()):
        rotation_wl, translation_wl = scene.sensor_trajectory.get_lidar_pose(i)
        rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl)
        rotation_np = rotation_wl.detach().cpu().numpy()
        translation_np = translation_wl.detach().cpu().numpy()
        pose_wl = np.eye(4)
        pose_wl[:3, 3] = translation_np
        pose_wl[:3, :3] = rotation_np
        lidar_poses_gt_matrices.append(pose_wl)
    pose_viz_thread.add_static_pose_group(
        lidar_poses_gt_matrices,
        frame_size=0.4, sphere_radius=0.04, sphere_color=[0, 0, 1]  # Blue
    )

    # == 相机位姿真值 (World to Camera) ==
    camera_poses_gt_matrices = []
    for i in range(scene.sensor_trajectory.get_length()):
        rotation_wc, translation_wc = scene.sensor_trajectory.get_gt_camera_pose(i)
        rotation_np = rotation_wc.detach().cpu().numpy()
        translation_np = translation_wc.detach().cpu().numpy()
        pose_wc = np.eye(4)
        pose_wc[:3, 3] = translation_np
        pose_wc[:3, :3] = rotation_np
        camera_poses_gt_matrices.append(pose_wc)
    pose_viz_thread.add_static_pose_group(
        camera_poses_gt_matrices,
        frame_size=0.5, sphere_radius=0.05, sphere_color=[0, 1, 0]  # Green
    )

    # --- 3. Main loop to update current camera pose ---
    # Example: Simulate a moving "current" camera pose (e.g., an estimate from an algorithm)
    # We'll base it on the first camera GT pose and perturb it.
    current_camera_poses_matrices = []
    for i in range(scene.sensor_trajectory.get_length()):
        rotation_wc, translation_wc = scene.sensor_trajectory.get_camera_pose(i)
        rotation_np = rotation_wc.detach().cpu().numpy()
        translation_np = translation_wc.detach().cpu().numpy()
        pose_wc = np.eye(4)
        pose_wc[:3, 3] = translation_np
        pose_wc[:3, :3] = rotation_np
        current_camera_poses_matrices.append(pose_wc)
    pose_viz_thread.update_camera_current_poses(current_camera_poses_matrices)
    return pose_viz_thread

def update_current_poses(scene, pose_viz_thread):
    current_camera_poses_matrices = []
    for i in range(scene.sensor_trajectory.get_length()):
        rotation_wc, translation_wc = scene.sensor_trajectory.get_camera_pose(i)
        rotation_np = rotation_wc.detach().cpu().numpy()
        translation_np = translation_wc.detach().cpu().numpy()
        pose_wc = np.eye(4)
        pose_wc[:3, 3] = translation_np
        pose_wc[:3, :3] = rotation_np
        current_camera_poses_matrices.append(pose_wc)
    pose_viz_thread.update_camera_current_poses(current_camera_poses_matrices)


# --- Main Program Logic ---
if __name__ == "__main__":
    # 1. Create and start the visualizer thread
    pose_viz_thread = PoseVisualizerThread()
    pose_viz_thread.start()

    # --- 2. Prepare and add static pose data ---
    # == LiDAR 位姿真值 (World to LiDAR) ==
    lidar_poses_gt_matrices = []
    for i in range(3):
        R_lidar = get_rotation_matrix_from_xyz_angles(0, 0, i * 10, in_degrees=True)
        t_lidar = np.array([i * 1.0, 0, 0])
        lidar_poses_gt_matrices.append(create_transformation_matrix(R_lidar, t_lidar))
    pose_viz_thread.add_static_pose_group(
        lidar_poses_gt_matrices,
        frame_size=0.4, sphere_radius=0.04, sphere_color=[0, 0, 1]  # Blue
    )

    # == 相机位姿真值 (World to Camera) ==
    R_cam_ext = get_rotation_matrix_from_xyz_angles(0, -90, 0, in_degrees=True)
    t_cam_ext = np.array([0.5, 0, 0])
    T_lidar_to_cam_extrinsic = create_transformation_matrix(R_cam_ext, t_cam_ext)
    camera_poses_gt_matrices = []
    for T_world_to_lidar in lidar_poses_gt_matrices:
        T_world_to_camera = T_world_to_lidar @ T_lidar_to_cam_extrinsic
        camera_poses_gt_matrices.append(T_world_to_camera)
    pose_viz_thread.add_static_pose_group(
        camera_poses_gt_matrices,
        frame_size=0.5, sphere_radius=0.05, sphere_color=[0, 1, 0]  # Green
    )

    # --- 3. Main loop to update current camera pose ---
    # Example: Simulate a moving "current" camera pose (e.g., an estimate from an algorithm)
    # We'll base it on the first camera GT pose and perturb it.
    if camera_poses_gt_matrices:
        base_cam_pose = camera_poses_gt_matrices[0].copy()
        current_cam_poses_to_show = []  # List to hold the current poses

        try:
            for step in range(200):  # Simulate 200 steps
                if not pose_viz_thread.is_alive():
                    print("Visualizer thread terminated unexpectedly.")
                    break

                # Create a new perturbed pose for the "current camera"
                # Simple oscillation example
                dx = 0.1 * np.sin(step * 0.1)
                dy = 0.05 * np.cos(step * 0.07)
                d_roll = 5 * np.sin(step * 0.05)  # degrees

                perturbed_pose = base_cam_pose.copy()
                perturbed_pose[0, 3] += dx  # Perturb x translation
                perturbed_pose[1, 3] += dy  # Perturb y translation

                R_perturb = get_rotation_matrix_from_xyz_angles(d_roll, 0, 0, in_degrees=True)
                perturbed_pose[:3, :3] = perturbed_pose[:3, :3] @ R_perturb  # Apply rotation perturbation

                current_cam_poses_to_show = [perturbed_pose]  # We are showing only one "current" pose

                # Update the visualizer with the new list of current camera poses
                pose_viz_thread.update_camera_current_poses(current_cam_poses_to_show)

                print(f"Step {step}: Updated current camera pose (dx={dx:.2f}, dy={dy:.2f}, d_roll={d_roll:.1f}).")
                time.sleep(0.1)  # Simulate work and update frequency

        except KeyboardInterrupt:
            print("Main thread interrupted by user.")
        finally:
            print("Stopping visualizer thread...")
            pose_viz_thread.stop()
            pose_viz_thread.join(timeout=5)  # Wait for the thread to finish
            if pose_viz_thread.is_alive():
                print("Warning: Visualizer thread did not stop gracefully.")
            print("Main program finished.")
    else:
        print("No camera GT poses to base current pose on. Exiting.")
        pose_viz_thread.stop()
        pose_viz_thread.join()

