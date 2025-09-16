from evo.tools import plot
from evo.core import lie_algebra as lie
from evo.tools.settings import SETTINGS
import matplotlib.pyplot as plt
from matplotlib import colormaps

import numpy as np
import rerun as rr
from scipy.spatial import transform
import torch
from rerun.components import Material
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

from src.processing.alignment import project_mesh, find_bounding_box

SETTINGS.plot_xyz_realistic = False


def _index_to_color(x, cmap="tab10"):
    return colormaps[cmap](x % colormaps[cmap].N)


def draw_trajectories(trajectories, save_path=None, marker_scale=0):
    plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
    # figsize = tuple(SETTINGS.plot_figsize)
    figsize = (15, 15)
    fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=figsize)
    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=figsize)
    fig_traj = plt.figure(figsize=figsize)
    plot_mode = plot.PlotMode[SETTINGS.plot_mode_default]
    ax_traj = plot.prepare_axis(fig_traj, plot_mode)

    for name, traj in trajectories.items():
        color = next(ax_traj._get_lines.prop_cycler)["color"]
        plot.traj_xyz(axarr_xyz, traj, color=color, label=name, alpha=0.5)
        plot.traj_rpy(axarr_rpy, traj, color=color, label=name, alpha=0.5)
        plot.traj(ax_traj, plot_mode, traj, color=color, label=name, alpha=0.5)
        # array = np.arange(len(traj.poses_se3)) / len(traj.poses_se3)
        # plot.traj_colormap(ax_traj, traj, array, plot_mode, min_map=0, max_map=1)
        plot.draw_coordinate_axes(ax_traj, traj, plot_mode, marker_scale=marker_scale)
    plot_collection.add_figure("trajectories", fig_traj)
    plot_collection.add_figure("xyz_view", fig_xyz)
    plot_collection.add_figure("rpy_view", fig_rpy)
    if save_path is not None:
        plot_collection.export(save_path, confirm_overwrite=True)

    return plot_collection


def log_trajectory(
    root_name,
    w2c_poses,
    c2w_poses,
    intrinsics,
    meshes=None,
    image_dir=None,
    main_characters=[],
):
    # Log trajectory
    rr.log(root_name, rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    rr.log(
        f"{root_name}/trajectory/points",
        rr.Points3D(c2w_poses[:, :3, 3]),
        timeless=True,
    )
    rr.log(
        f"{root_name}/trajectory/line",
        rr.LineStrips3D(
            np.stack((c2w_poses[:, :3, 3][:-1], c2w_poses[:, :3, 3][1:]), axis=1),
            colors=[(1.0, 0.0, 1.0, 1.0)],
        ),
        timeless=True,
    )

    if meshes is not None:
        vertices, faces = meshes
        num_tracks, num_frames, num_verts, _ = vertices.shape
        _vertices = vertices.clone().reshape(num_tracks * num_frames, num_verts, 3)
        # Compute normals based on aligned aligned_verts
        faces = faces.expand(_vertices.shape[0], faces.shape[-2], 3)
        textures = TexturesVertex(verts_features=torch.ones_like(_vertices))
        meshes = Meshes(verts=_vertices, faces=faces, textures=textures)
        normals = meshes.verts_normals_padded()

        num_tracks = vertices.shape[0]
        colors = [
            (0.0, 0.0, 0.0, 1.0) if i not in main_characters else _index_to_color(i)
            for i in range(num_tracks)
        ]
        annotation_list = [
            rr.AnnotationInfo(id=i + 1, label="human_{i+1}", color=colors[i])
            for i in range(num_tracks)
        ]
        rr.log(
            f"{root_name}/camera/image/mask",
            rr.AnnotationContext(annotation_list),
            timeless=True,
        )

    num_cameras = c2w_poses.shape[0]
    for k in range(num_cameras):
        rr.set_time_sequence("frame_idx", k)
        if image_dir is not None:
            rr.log(
                f"{root_name}/camera/image",
                rr.ImageEncoded(path=image_dir / f"{str(k+1).zfill(5)}.jpg"),
            )

        fx, fy, cx, cy = intrinsics[k]
        translation = c2w_poses[k][:3, 3]
        rotation_q = transform.Rotation.from_matrix(c2w_poses[k][:3, :3]).as_quat()
        rr.log(
            f"{root_name}/camera/image",
            rr.Pinhole(
                image_from_camera=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                width=cx * 2,
                height=cy * 2,
            ),
        )
        rr.log(
            f"{root_name}/camera",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_q),
            ),
        )
        rr.log(
            f"{root_name}/pov/image",
            rr.Pinhole(
                image_from_camera=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                width=cx * 2,
                height=cy * 2,
            ),
        )
        rr.log(
            f"{root_name}/pov",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_q),
            ),
        )
        rr.set_time_sequence("image", k)

        if meshes is not None:
            K = torch.eye(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            Rt = torch.eye((4))
            Rt[:3, :3] = w2c_poses[k][:3, :3]
            Rt[:3, 3] = w2c_poses[k][:3, 3]
            height, width = int(cy * 2), int(cx * 2)
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            depth_mask = np.inf * np.ones((height, width), dtype=np.float32)
            M = K @ w2c_poses[k][:3, :3]
            m = np.array(M[2]) / np.linalg.norm(M[2])
            num_tracks = vertices.shape[0]
            for i in range(num_tracks):
                d = (vertices[i, k] - c2w_poses[k, :3, 3].numpy()) @ m
                # Mesh behind the camera
                if d.max() < 1:
                    rr.log(f"{root_name}/human/human_{i}", rr.Clear(recursive=False))
                    rr.log(
                        f"{root_name}/camera/image/bbox_{i}", rr.Clear(recursive=False)
                    )
                    continue
                # Null vertices
                if vertices[i, k].sum() == 0:
                    rr.log(f"{root_name}/human/human_{i}", rr.Clear(recursive=False))
                    rr.log(
                        f"{root_name}/camera/image/bbox_{i}", rr.Clear(recursive=False)
                    )
                    continue
                verts_h = np.hstack(
                    (vertices[i, k], np.ones((vertices[i, k].shape[0], 1)))
                )
                mask = project_mesh(verts_h, faces[0], Rt, K, height, width) * (i + 1)
                mesh_depth = np.mean(verts_h, axis=0)[2]
                depth_mask[mask > 0] = np.where(
                    mesh_depth <= depth_mask[mask > 0], mesh_depth, depth_mask[mask > 0]
                )
                combined_mask[mask > 0] = np.where(
                    mesh_depth <= depth_mask[mask > 0],
                    mask[mask > 0],
                    combined_mask[mask > 0],
                )
                rr.log(
                    f"{root_name}/human/human_{i}",
                    rr.Mesh3D(
                        vertex_positions=vertices[i, k],
                        indices=faces[0],
                        vertex_normals=normals[i, k],
                        mesh_material=Material(albedo_factor=colors[i]),
                    ),
                )
                if image_dir is not None:
                    rr.log(
                        f"{root_name}/camera/image/bbox_{i}",
                        rr.Boxes2D(
                            array=find_bounding_box(mask),
                            array_format=rr.Box2DFormat("XYXY"),
                            colors=colors[i],
                        ),
                    )
            if image_dir is not None:
                rr.log(
                    f"{root_name}/camera/image/mask",
                    rr.SegmentationImage(combined_mask),
                )


def log_dynamics(c2w_poses):
    # TODO: to fix
    raw_trans = c2w_poses[:, :3, 3]
    raw_rotations = c2w_poses[:, :3, :3]
    translations = raw_trans - raw_trans[0]
    # for r in raw_rotations:
    #     rotations = np.stack([lie.so3_log(r.numpy()) for r in raw_rotations])
    t_velocities = torch.concatenate(
        [torch.zeros((1, 3)), 24 * (translations[:-1] - translations[1:])]
    )

    # a_velocities = torch.from_numpy(
    #     np.concatenate([np.zeros((1, 3)), 24 * (rotations[:-1] - rotations[1:])])
    # )
    t_accelerations = t_velocities[:-1] - t_velocities[1:]
    # a_accelerations = a_velocities[:-1] - a_velocities[1:]
    t_jerks = t_accelerations[:-1] - t_accelerations[1:]
    # a_jerks = a_accelerations[:-1] - a_accelerations[1:]

    num_frames = len(translations)
    x_velocity = np.pad(
        t_velocities.numpy()[:, 0],
        (0, num_frames - t_velocities.shape[0]),
        mode="edge",
    )
    y_velocity = np.pad(
        t_velocities.numpy()[:, 1],
        (0, num_frames - t_velocities.shape[0]),
        mode="edge",
    )
    z_velocity = np.pad(
        t_velocities.numpy()[:, 2],
        (0, num_frames - t_velocities.shape[0]),
        mode="edge",
    )
    t_velocity = np.pad(
        torch.linalg.norm(t_velocities, dim=1).numpy(),
        (0, num_frames - t_velocities.shape[0]),
        mode="edge",
    )
    t_acceleration = np.pad(
        torch.linalg.norm(t_accelerations, dim=1).numpy(),
        (0, num_frames - t_accelerations.shape[0]),
        mode="edge",
    )
    t_jerk = np.pad(
        torch.linalg.norm(t_jerks, dim=1).numpy(),
        (0, num_frames - t_jerks.shape[0]),
        mode="edge",
    )
    # a_velocity = np.pad(
    #     np.linalg.norm(a_velocities, axis=1),
    #     (0, num_frames - a_velocities.shape[0]),
    #     mode="edge",
    # )
    # a_acceleration = np.pad(
    #     np.linalg.norm(a_accelerations, axis=1),
    #     (0, num_frames - a_accelerations.shape[0]),
    #     mode="edge",
    # )
    # a_jerk = np.pad(
    #     np.linalg.norm(a_jerks, axis=1),
    #     (0, num_frames - a_jerks.shape[0]),
    #     mode="edge",
    # )

    for k in range(num_frames):
        rr.set_time_sequence("frame_idx", k)

        # Translation
        root_name = "translations"
        rr.log(
            f"{root_name}/x",
            rr.TimeSeriesScalar(translations[k, 0], color=_index_to_color(0)),
        )
        rr.log(
            f"{root_name}/y",
            rr.TimeSeriesScalar(translations[k, 1], color=_index_to_color(0)),
        )
        rr.log(
            f"{root_name}/z",
            rr.TimeSeriesScalar(translations[k, 2], color=_index_to_color(0)),
        )
        rr.log(
            f"{root_name}/x_velocity",
            rr.TimeSeriesScalar(x_velocity[k], color=_index_to_color(3)),
        )
        rr.log(
            f"{root_name}/y_velocity",
            rr.TimeSeriesScalar(y_velocity[k], color=_index_to_color(4)),
        )
        rr.log(
            f"{root_name}/z_velocity",
            rr.TimeSeriesScalar(z_velocity[k], color=_index_to_color(5)),
        )
        rr.log(
            f"{root_name}/velocity",
            rr.TimeSeriesScalar(t_velocity[k], color=_index_to_color(1)),
        )
        rr.log(
            f"{root_name}/acceleration",
            rr.TimeSeriesScalar(t_acceleration[k], color=_index_to_color(3)),
        )
        rr.log(
            f"{root_name}/jerk",
            rr.TimeSeriesScalar(t_jerk[k], color=_index_to_color(2)),
        )

        # # Rotation
        # root_name = "rotations"
        # rr.log(
        #     f"{root_name}/r",
        #     rr.TimeSeriesScalar(rotations[k, 0], color=_index_to_color(0)),
        # )
        # rr.log(
        #     f"{root_name}/p",
        #     rr.TimeSeriesScalar(rotations[k, 1], color=_index_to_color(0)),
        # )
        # rr.log(
        #     f"{root_name}/y",
        #     rr.TimeSeriesScalar(rotations[k, 2], color=_index_to_color(0)),
        # )
        # rr.log(
        #     f"{root_name}/velocity",
        #     rr.TimeSeriesScalar(a_velocity[k], color=_index_to_color(1)),
        # )
        # rr.log(
        #     f"{root_name}/acceleration",
        #     rr.TimeSeriesScalar(a_acceleration[k], color=_index_to_color(3)),
        # )
        # rr.log(
        #     f"{root_name}/jerk",
        #     rr.TimeSeriesScalar(a_jerk[k], color=_index_to_color(2)),
        # )
