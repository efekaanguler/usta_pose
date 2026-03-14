
import json
import numpy as np
import cv2


def load_camera_serials(config_path):
    """Load camera serial numbers from a JSON config file.

    Expected format: {"cam1": "serial1", "cam2": "serial2", ...}
    Returns dict mapping camera ID (int) to serial string.
    """
    with open(config_path) as f:
        data = json.load(f)
    serials = {}
    for key, value in data.items():
        key_lower = key.strip().lower()
        if key_lower.startswith('cam'):
            try:
                cam_id = int(key_lower[3:])
                serials[cam_id] = str(value)
            except ValueError:
                continue
    return serials

def solve_projection_matrix(pts3d, pts2d):
    """
    Solve for projection matrix P using Direct Linear Transform (DLT)
    Args:
        pts3d: (N, 3) array of 3D joint coordinates (in local coordinate system)
        pts2d: (N, 2) array of corresponding 2D image coordinates
    Returns:
        P: (3, 4) projection matrix
    """
    assert len(pts3d) == len(pts2d), "Number of 2D and 3D points must match"
    assert len(pts3d) >= 6, "Need at least 6 points for DLT"

    # Build matrix A for homogeneous system Ap = 0
    A = []
    for i in range(len(pts3d)):
        X, Y, Z = pts3d[i]
        u, v = pts2d[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    # Solve using SVD (last column of V)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    # P /= P[2,3]  # Normalize
    
    return P

def decompose_projection_matrix(P):
    """
    Decompose projection matrix P into K, R, and t using RQ decomposition
    Returns:
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    """
    # Extract left 3x3 submatrix and perform RQ decomposition
    M = P[:, :3]
    
    # Use QR decomposition for RQ (reverse order using permutation matrix)
    H = np.eye(3)[::-1]
    Q, R = np.linalg.qr(H @ M.T @ H)
    
    # Recover K and R
    K = H @ R.T @ H
    R = H @ Q.T @ H
    
    # Ensure positive diagonal for K
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # Normalize K (make K[2,2] = 1)
    K /= K[2,2]
    
    # Solve for translation vector t = K^-1 * P[:,3]
    t = np.linalg.inv(K) @ P[:,3]
    
    # Ensure proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        R *= -1
        t *= -1
    
    return K, R, t

def decompose_projection_matrix_with_fixed_intrinsics(P, fx, fy, cx, cy):
    """
    Decompose P into R and t, assuming fixed intrinsics K.
    Args:
        P: (3, 4) projection matrix
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels
    Returns:
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    """
    # Construct fixed intrinsic matrix K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Solve for extrinsics [R | t] = K^-1 P
    K_inv = np.linalg.inv(K)
    Rt = K_inv @ P
    
    # Extract R and t
    R = Rt[:, :3]
    t = Rt[:, 3]
    
    # Ensure R is a valid rotation matrix (orthogonal with det(R) = 1)
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R *= -1
        t *= -1
    
    return R, t

def transform_to_camera_frame(pts3d_local, R, t):
    """
    Transform local 3D points to camera coordinate system
    Args:
        pts3d_local: (N, 3) array of local 3D points
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    Returns:
        pts3d_cam: (N, 3) points in camera coordinates
    """
    
    return (R @ pts3d_local.T + t.reshape(-1, 1)).T


def vectorized_transform_to_camera_frame(pts3d_local, R, t):
    """
    Transform local 3D points to camera coordinate system
    Args:
        pts3d_local: (N, 3) array of local 3D points
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    Returns:
        pts3d_cam: (N, 3) points in camera coordinates
    """
    return np.matmul(pts3d_local, R.transpose(0, 2, 1)) + t[:, None, :]


def local_to_camera_transformation(pts3d, pts2d, fx, fy, cx, cy, dist_coeffs=None):

    # Construct fixed intrinsic matrix K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Distortion coefficients (assuming zero for now)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)

    # SolvePnP to get rotation and translation
    success, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

    if not success:
        print("Error: Failed to solve PnP")
        return

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec


def local_to_camera_transformation_ransac(pts3d, pts2d, fx, fy, cx, cy, dist_coeffs=None):

    # Construct fixed intrinsic matrix K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Distortion coefficients (assuming zero for now)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)

    # SolvePnP to get rotation and translation
    success, rvec, tvec, _ = cv2.solvePnPRansac(pts3d, pts2d, K, dist_coeffs)

    # if not success:
    #     print("Error: Failed to solve PnP")
    #     return
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec


def smoothing_factor(t_e: float, cutoff: np.ndarray) -> np.ndarray:
    """
    Compute smoothing factor α for time elapsed t_e and cutoff frequency(s).
    α = (2π·cutoff·t_e) / (1 + 2π·cutoff·t_e)
    """
    r = 2 * np.pi * cutoff * t_e
    return r / (1.0 + r)


def exponential_smoothing(alpha: np.ndarray, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    """
    Perform one step of exponential smoothing:
        x_hat = α·x + (1−α)·x_prev
    """
    return alpha * x + (1.0 - alpha) * x_prev


class OneEuroFilterVector:
    """
    One Euro filter, vectorized over arbitrary NumPy array shapes.

    Parameters
    ----------
    t0 : float
        Initial timestamp.
    x0 : np.ndarray
        Initial signal array (e.g. shape (33,3) for 33 joints in 3D).
    dx0 : np.ndarray, optional
        Initial derivative array (same shape as x0). Default is zeros.
    min_cutoff : float, optional
        Minimum cutoff frequency. Default is 0.05.
    beta : float, optional
        Speed-bias factor. Default is 80.0.
    d_cutoff : float, optional
        Cutoff frequency for the derivative. Default is 1.0.
    """
    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        dx0: np.ndarray = None,
        min_cutoff: float = 0.05,
        beta: float = 80.0,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        # Previous state (arrays of same shape as x0)
        self.x_prev = x0.astype(float)
        self.dx_prev = np.zeros_like(self.x_prev) if dx0 is None else dx0.astype(float)
        self.t_prev = float(t0)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Apply the filter to a new sample x at time t.

        Parameters
        ----------
        t : float
            Current timestamp.
        x : np.ndarray
            Current signal array (same shape as x0).

        Returns
        -------
        np.ndarray
            The filtered signal (same shape as x).
        """
        t_e = t - self.t_prev
        if t_e <= 0:
            # No time elapsed: return previous value
            return self.x_prev

        # 1) Derivative of the signal
        dx = (x - self.x_prev) / t_e

        # 2) Filter derivative
        a_d = smoothing_factor(t_e, self.d_cutoff)  # scalar → broadcast to array
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # 3) Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # 4) Filter signal
        a = smoothing_factor(t_e, cutoff)  # array of shape x
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat