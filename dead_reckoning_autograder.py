import os
import sys
import argparse
import numpy as np

import dead_reckoning_submission as student


# --------------------------
# HARD-CODED GRADING PARAMS
# --------------------------
WHEEL_RADIUS = 0.1
WHEEL_BASE   = 0.2
ENC_RES      = 4096
GRADE_RATE_HZ = 50
MAX_POS_ERR_M_THRESH = 0.028    # meters
MAX_THETA_ERR_RAD_THRESH = 0.02 # radians
CHECK_THETA_TOO = True


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def interp_to(t_src, x_src, t_dst):
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


def pick_stream(encLeft2, encRight2, encLeft10, encRight10, encLeft50, encRight50, rate_hz: int):
    if rate_hz == 2:
        return encLeft2, encRight2, 1.0 / 2.0
    if rate_hz == 10:
        return encLeft10, encRight10, 1.0 / 10.0
    if rate_hz == 50:
        return encLeft50, encRight50, 1.0 / 50.0
    raise ValueError("rate_hz must be one of {2,10,50}")


def grade(npz_path: str):
    encLeft2, encRight2, encLeft10, encRight10, encLeft50, encRight50, xRGT, yRGT, thetaGT, tGT = load_npz_data(npz_path)

    leftEnc, rightEnc, dt = pick_stream(encLeft2, encRight2, encLeft10, encRight10, encLeft50, encRight50, GRADE_RATE_HZ)
    
    try:
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
    except NotImplementedError as e:
        return False, {"error": f"NotImplementedError: {e}"}
    except Exception as e:
        return False, {"error": f"Exception while running student code: {repr(e)}"}

    # Basic sanity checks
    t_enc = np.asarray(t_enc, dtype=float)
    x_est = np.asarray(x_est, dtype=float)
    y_est = np.asarray(y_est, dtype=float)
    th_est = np.asarray(th_est, dtype=float)

    if len(t_enc) != len(x_est) or len(x_est) != len(y_est) or len(y_est) != len(th_est):
        return False, {"error": "Output arrays t,x,y,th must have the same length."}

    if len(t_enc) < 2:
        return False, {"error": "Output arrays are too short."}

    # Interpolate onto GT timestamps for comparison
    x_est_gt = interp_to(t_enc, x_est, tGT)
    y_est_gt = interp_to(t_enc, y_est, tGT)
    th_est_gt = interp_to(t_enc, th_est, tGT)

    # Compute errors
    pos_err = np.sqrt((x_est_gt - xRGT) ** 2 + (y_est_gt - yRGT) ** 2)
    max_pos_err = float(np.max(pos_err))

    theta_err = wrap_to_pi(th_est_gt - thetaGT)
    max_theta_err = float(np.max(np.abs(theta_err)))

    passed = (max_pos_err <= MAX_POS_ERR_M_THRESH)
    if CHECK_THETA_TOO:
        passed = passed and (max_theta_err <= MAX_THETA_ERR_RAD_THRESH)

    info = {
        "rate_hz": GRADE_RATE_HZ,
        "max_pos_err_m": max_pos_err,
        "pos_thresh_m": MAX_POS_ERR_M_THRESH,
        "max_theta_err_rad": max_theta_err,
        "theta_thresh_rad": MAX_THETA_ERR_RAD_THRESH,
        "check_theta_too": CHECK_THETA_TOO,
    }
    return passed, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--npz",
    type=str,
    default="assignment_1/assets/ps1_encoderData.npz",
    help="Path to encoder + GT data (.npz)"
    )
    args = parser.parse_args()
    print(f"[AUTOGRADER] Loading data from: {args.npz}")

    passed, info = grade(args.npz)

    if not passed:
        print("FAIL")
        for k, v in info.items():
            print(f"{k}: {v}")
        sys.exit(1)

    print("PASS")
    for k, v in info.items():
        print(f"{k}: {v}")
    sys.exit(0)


if __name__ == "__main__":
    main()
