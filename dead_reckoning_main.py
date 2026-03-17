import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

import dead_reckoning_submission as student


# --------------------------
# Robot parameters (given)
# --------------------------
WHEEL_RADIUS = 0.1
WHEEL_BASE   = 0.2
ENC_RES      = 4096


# --------------------------
# Helpers
# --------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_quat(th: float) -> np.ndarray:
    return np.array([np.cos(th / 2.0), 0.0, 0.0, np.sin(th / 2.0)], dtype=float)


def set_body_pose_freejoint(data: mujoco.MjData, qpos_adr: int, x: float, y: float, z: float, yaw: float):
    q = yaw_to_quat(yaw)
    data.qpos[qpos_adr + 0] = x
    data.qpos[qpos_adr + 1] = y
    data.qpos[qpos_adr + 2] = z
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = q


def get_freejoint_qpos_adr(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in XML.")

    jnt_adr = model.body_jntadr[body_id]
    if jnt_adr < 0:
        raise ValueError(f"Body '{body_name}' has no joint. Add <freejoint/> in XML.")

    return int(model.jnt_qposadr[jnt_adr])


def interp_to(t_src: np.ndarray, x_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.interp(t_dst, t_src, x_src)


def load_npz_data(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    tmp = np.load(npz_path)
    encLeft2  = tmp["encLeft2"]
    encRight2 = tmp["encRight2"]
    encLeft10  = tmp["encLeft10"]
    encRight10 = tmp["encRight10"]
    encLeft50  = tmp["encLeft50"]
    encRight50 = tmp["encRight50"]

    xRGT = tmp["xRGT"]
    yRGT = tmp["yRGT"]
    thetaGT = tmp["thetaGT"]
    tGT = tmp["tGT"]
    return encLeft2, encRight2, encLeft10, encRight10, encLeft50, encRight50, xRGT, yRGT, thetaGT, tGT


# --------------------------
# Optional noise (for visualization/demo)
# --------------------------
def add_encoder_noise(
    encLeft: np.ndarray,
    encRight: np.ndarray,
    *,
    sigma_ticks_per_step: float = 3.0,
    bias_ticks_per_step: float = 0.15,
    slip_prob: float = 0.03,
    slip_scale_std: float = 0.10,
    seed: int = 42,
):
    """
    Adds noise to encoder increments only (no model mismatch here).
    Autograder should typically run WITHOUT noise.
    """
    rng = np.random.default_rng(seed)
    encLeft = np.asarray(encLeft, dtype=float)
    encRight = np.asarray(encRight, dtype=float)

    dL = np.diff(encLeft)
    dR = np.diff(encRight)

    dL_noisy = dL + rng.normal(0.0, sigma_ticks_per_step, size=dL.shape) + bias_ticks_per_step
    dR_noisy = dR + rng.normal(0.0, sigma_ticks_per_step, size=dR.shape) - bias_ticks_per_step

    slip_mask_L = rng.random(dL.shape) < slip_prob
    slip_mask_R = rng.random(dR.shape) < slip_prob

    slip_factor_L = 1.0 + rng.normal(0.0, slip_scale_std, size=dL.shape)
    slip_factor_R = 1.0 + rng.normal(0.0, slip_scale_std, size=dR.shape)

    dL_noisy = np.where(slip_mask_L, dL_noisy * slip_factor_L, dL_noisy)
    dR_noisy = np.where(slip_mask_R, dR_noisy * slip_factor_R, dR_noisy)

    encLeft_noisy = np.empty_like(encLeft)
    encRight_noisy = np.empty_like(encRight)
    encLeft_noisy[0] = encLeft[0]
    encRight_noisy[0] = encRight[0]
    encLeft_noisy[1:] = encLeft_noisy[0] + np.cumsum(dL_noisy)
    encRight_noisy[1:] = encRight_noisy[0] + np.cumsum(dR_noisy)
    return encLeft_noisy, encRight_noisy


# --------------------------
# Plotting
# --------------------------
def plot_gt_vs_est(tGT, xRGT, yRGT, thetaGT, x_est_gt, y_est_gt, th_est_gt):
    pos_err = np.sqrt((x_est_gt - xRGT) ** 2 + (y_est_gt - yRGT) ** 2)
    th_err = wrap_to_pi(th_est_gt - thetaGT)

    plt.figure(figsize=(7, 6))
    plt.plot(xRGT, yRGT, linewidth=2, label="Ground Truth")
    plt.plot(x_est_gt, y_est_gt, "--", linewidth=2, label="Dead Reckoning")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("XY Trajectory")
    plt.legend()

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tGT, xRGT, linewidth=2, label="GT")
    ax1.plot(tGT, x_est_gt, "--", linewidth=2, label="Est")
    ax1.set_ylabel("x [m]")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(tGT, yRGT, linewidth=2)
    ax2.plot(tGT, y_est_gt, "--", linewidth=2)
    ax2.set_ylabel("y [m]")
    ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(tGT, wrap_to_pi(thetaGT), linewidth=2)
    ax3.plot(tGT, wrap_to_pi(th_est_gt), "--", linewidth=2)
    ax3.set_ylabel("theta [rad]")
    ax3.set_xlabel("time [s]")
    ax3.grid(True)

    plt.figure(figsize=(10, 6))
    ax4 = plt.subplot(2, 1, 1)
    ax4.plot(tGT, pos_err, linewidth=2)
    ax4.set_ylabel("pos err [m]")
    ax4.grid(True)
    ax4.set_title("Error")

    ax5 = plt.subplot(2, 1, 2, sharex=ax4)
    ax5.plot(tGT, th_err, linewidth=2)
    ax5.set_ylabel("theta err [rad]")
    ax5.set_xlabel("time [s]")
    ax5.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--npz",
    type=str,
    default="assignment_1/assets/ps1_encoderData.npz",
    help="Path to encoder + GT data (.npz)"
    )
    parser.add_argument(
    "--xml",
    type=str,
    default="assignment_1/assets/two_robots_diffdrive.xml",
    help="Path to MuJoCo XML"
    )
    parser.add_argument("--rate", type=int, default=50, choices=[2, 10, 50])
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action="store_true")

    # Visualization noise toggles (NOT for grading)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma_ticks", type=float, default=3.0)
    parser.add_argument("--bias_ticks", type=float, default=0.15)
    parser.add_argument("--slip_prob", type=float, default=0.03)
    parser.add_argument("--slip_std", type=float, default=0.10)

    # MuJoCo visualization options
    parser.add_argument("--z_est", type=float, default=0.12, help="Estimated robot height")
    parser.add_argument("--z_gt_offset", type=float, default=0.03, help="Ghost robot extra height offset")
    parser.add_argument("--cam_dist", type=float, default=8.0, help="Top-down camera distance")
    parser.add_argument("--follow_cam", action="store_true", help="Camera follows robots")

    args = parser.parse_args()
    print(f"[INFO] Using data file: {args.npz}")
    print(f"[INFO] Using MuJoCo XML: {args.xml}")

    # Load data
    encLeft2, encRight2, encLeft10, encRight10, encLeft50, encRight50, xRGT, yRGT, thetaGT, tGT = load_npz_data(args.npz)

    if args.rate == 2:
        leftEnc, rightEnc, dt = encLeft2, encRight2, 1.0 / 2.0
    elif args.rate == 10:
        leftEnc, rightEnc, dt = encLeft10, encRight10, 1.0 / 10.0
    else:
        leftEnc, rightEnc, dt = encLeft50, encRight50, 1.0 / 50.0

    # Optional noise for demo
    if args.noisy:
        leftEnc, rightEnc = add_encoder_noise(
            leftEnc, rightEnc,
            sigma_ticks_per_step=args.sigma_ticks,
            bias_ticks_per_step=args.bias_ticks,
            slip_prob=args.slip_prob,
            slip_scale_std=args.slip_std,
            seed=args.seed,
        )
        print("[INFO] Noise enabled for visualization.")

    # Call student code
    t_enc, x_est, y_est, th_est, vf, omega = student.dead_reckoning_from_encoders(
        leftEnc=leftEnc,
        rightEnc=rightEnc,
        dt=dt,
        encoderResolution=ENC_RES,
        wheelRadius=WHEEL_RADIUS,
        wheelBase=WHEEL_BASE,
        x0=float(xRGT[0]),
        y0=float(yRGT[0]),
        th0=float(thetaGT[0]),
    )

    # Interpolate estimate onto GT timestamps for synchronized plotting + animation
    x_est_gt = interp_to(t_enc, x_est, tGT)
    y_est_gt = interp_to(t_enc, y_est, tGT)
    th_est_gt = interp_to(t_enc, th_est, tGT)

    if args.plots:
        plot_gt_vs_est(tGT, xRGT, yRGT, thetaGT, x_est_gt, y_est_gt, th_est_gt)

    # MuJoCo
    if not os.path.exists(args.xml):
        raise FileNotFoundError(f"XML not found: {args.xml}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    gt_adr = get_freejoint_qpos_adr(model, "robot_gt")
    est_adr = get_freejoint_qpos_adr(model, "robot_est")

    z_est = float(args.z_est)
    z_gt = z_est + float(args.z_gt_offset)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # ---- TOP-DOWN CAMERA ----
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0
        viewer.cam.distance = float(args.cam_dist)
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])

        for k in range(len(tGT)):
            # Place GT and estimated
            set_body_pose_freejoint(data, gt_adr,  float(xRGT[k]),     float(yRGT[k]),     z_gt,  float(thetaGT[k]))
            set_body_pose_freejoint(data, est_adr, float(x_est_gt[k]), float(y_est_gt[k]), z_est, float(th_est_gt[k]))

            mujoco.mj_forward(model, data)

            if args.follow_cam:
                cx = 0.5 * (float(xRGT[k]) + float(x_est_gt[k]))
                cy = 0.5 * (float(yRGT[k]) + float(y_est_gt[k]))
                viewer.cam.lookat[:] = np.array([cx, cy, 0.0])

                # keep it locked top-down even if user drags mouse
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -90.0

            viewer.sync()

            if args.realtime and k < len(tGT) - 1:
                time.sleep(max(0.0, float(tGT[k + 1] - tGT[k])))

    print("Done.")


if __name__ == "__main__":
    main()
