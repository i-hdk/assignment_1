import numpy as np


def ticks2vel(leftEnc, rightEnc, encoderResolution, timeStep, wheelRadius, wheelBase):
    """
    Convert accumulated encoder ticks into robot forward velocity and yaw rate.

    Inputs
    ------
    leftEnc, rightEnc : np.ndarray, shape (N,)
        Accumulated encoder ticks for left/right wheel.
    encoderResolution : int
        Ticks per wheel revolution (e.g., 4096).
    timeStep : float
        Sampling period in seconds.
    wheelRadius : float
        Wheel radius in meters (e.g., 0.1).
    wheelBase : float
        Distance between wheel centers in meters (e.g., 0.2).

    Returns
    -------
    vf : np.ndarray, shape (N-1,)
        Forward velocity of robot (m/s) over each interval k -> k+1.
    omega : np.ndarray, shape (N-1,)
        Yaw rate of robot (rad/s) over each interval k -> k+1.
    """
    N = len(leftEnc)

    # 1. Encoder tick increments (ticks per timestep)
    # From the Numpy documentation, "diff" computes and returns the difference:
    #    out[i] = in[i+1] - in[i]
    # for every element in the input argument "in." Thus leftTicks and rightTicks
    # will be size N-1
    leftTicks = np.diff(leftEnc)
    rightTicks = np.diff(rightEnc)

    # 2. Convert ticks -> wheel angular displacement (rad)
    #    2*pi rad per revolution encoderResolution ticks per revolution
    #
    # Note: if you are new to numpy, dphi_l will be an array of size N-1 
    # that results from multiplying (2pi/res) by every element in leftTicks
    # If unfamiliar, use a print(dphi_l) to see what its doing. This is
    # much faster, but equivalent to:
    #   for i in range(N-1):
    #      phi_l[i] = (2.0 * np.pi / encoderResolution) * leftTicks[i]
    dphi_l = (2.0 * np.pi / encoderResolution) * leftTicks
    dphi_r = (2.0 * np.pi / encoderResolution) * rightTicks

    # 3. Compute the wheel angular velocity (rad/s)
    #   dphi_l and dphi_r contain the angular displacement between two timesteps
    #   We want omega_l and omega_r to be expressed in radians/seconds
    # Hint: you can use a loop, but you can also the same trick we did
    # to compute dphi_l and dphi_r.
    omega_l = dphi_l / timeStep
    omega_r = dphi_r / timeStep

    # 4. Wheel linear velocities (m/s)
    # Now that you have the angular velocities, convert these to linear velocities (m/s)
    v_l = omega_l * wheelRadius
    v_r = omega_r * wheelRadius

    # 5. Differential drive kinematics
    # TODO: compute vf (velocity forward) and omega (angular velocity) based on values above 
    # (replacing the plaeholder assignments below.)
    vf = (v_l + v_r) / 2.0
    omega = (v_r - v_l) / wheelBase

    return vf, omega

def dead_reckoning_from_encoders(leftEnc, rightEnc, dt, encoderResolution, wheelRadius, wheelBase, x0, y0, th0):
    """
    Dead-reckon robot pose from encoder ticks.

    Inputs
    ------
    leftEnc, rightEnc : np.ndarray, shape (N,)
        Accumulated ticks.
    dt : float
        Sampling period (seconds).
    encoderResolution : int
        ticks/rev
    wheelRadius : float
        meters
    wheelBase : float
        meters
    x0, y0, th0 : float
        initial pose

    Returns
    -------
    t : np.ndarray, shape (N,)
        timestamps starting at 0
    x, y, th : np.ndarray, shape (N,)
        estimated pose at each sample time (aligned with encoders)
    vf, omega : np.ndarray, shape (N-1,)
        estimated body velocities for each interval
    """

    # 0. Setup (and sanity check)
    N = len(leftEnc)
    if len(rightEnc) != N:
        raise ValueError("leftEnc and rightEnc must have same length")

    # 1. Compute body velocities (Do not modify here, implement method above)
    # NOTE: ticks2vel must work correctly for your loop to work below
    vf, omega = ticks2vel(
        leftEnc, rightEnc,
        encoderResolution, dt,
        wheelRadius, wheelBase)

    # 2. Compute the time vector (do not modify)
    t = np.arange(N) * dt

    # 3. Allocate the state arrays
    x = np.zeros(N)
    y = np.zeros(N)
    th = np.zeros(N)

    # Initial condition
    x[0] = x0
    y[0] = y0
    th[0] = th0

    # 4. Integrate pose, write your code here:
    for k in range(1,N):
        # TODO: compute th[k] based on th[k-1] and the direction and velocity
        th[k] = th[k-1] + omega[k-1] * dt
        # TODO: compute x[k] based on x[k-1] and the direction and velocity
        x[k] = x[k-1] + vf[k-1] * np.cos(th[k-1]) * dt
        # TODO: compute y[k] based on y[k-1] and the direction and velocity
        y[k] = y[k-1] + vf[k-1] * np.sin(th[k-1]) * dt

    return t, x, y, th, vf, omega
    
