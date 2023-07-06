from kornia.core import Tensor
from kornia.geometry.vector import Vector2


class AffineTransform:
    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        """Distort one or more Vector2 points using the affine transform.

        Args:
            params: Tensor representing the affine transform parameters.
            points: Vector2 representing the points to distort.

        Returns:
            Vector2 representing the distorted points.

        Example:
            >>> params = Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().distort(params, points)
            x: 4.0
            y: 8.0
        """
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        u = points.x * fx + cx
        v = points.y * fy + cy
        return Vector2.from_coords(u, v)

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        """Undistort one or more Vector2 points using the affine transform.

        Args:
            params: Tensor representing the affine transform parameters.
            points: Vector2 representing the points to undistort.

        Returns:
            Vector2 representing the undistorted points.

        Example:
            >>> params = Tensor([1., 2., 3., 4.])
            >>> points = Vector2.from_coords(1., 2.)
            >>> AffineTransform().undistort(params, points)
            x: -2.0
            y: -1.0
        """
        fx, fy, cx, cy = params[..., 0], params[..., 1], params[..., 2], params[..., 3]
        x = (points.x - cx) / fx
        y = (points.y - cy) / fy
        return Vector2.from_coords(x, y)


class BrownConradyTransform:
    def project(self, distortion_params: Tensor, points: Vector2) -> Vector2:
        x, y = points.x, points.y
        k1, k2, p1, p2 = (
            distortion_params[..., 0],
            distortion_params[..., 1],
            distortion_params[..., 2],
            distortion_params[..., 3],
        )
        k3, k4, k5, k6 = (
            distortion_params[..., 4],
            distortion_params[..., 5],
            distortion_params[..., 6],
            distortion_params[..., 7],
        )
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * x * y
        a2 = r2 + 2 * x * x
        a3 = r2 + 2 * y * y
        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        icdist = 1 / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        xd0 = x * cdist * icdist + p1 * a1 + p2 * a2
        yd0 = y * cdist * icdist + p1 * a3 + p2 * a1
        return Vector2.from_coords(xd0, yd0)

    def unproject(self, distortion_params: Tensor, uv_normalized: Vector2) -> Vector2:
        xy = uv_normalized.data
        for i in range(50):
            x = xy[..., 0]
            y = xy[..., 1]
            f_xy = self.project(distortion_params, Vector2.from_coords(x, y)) - uv_normalized

            a = x
            b = y
            d = distortion_params

            c0 = a * a
            c1 = b * b
            c2 = c0 + c1
            c3 = c2 * c2
            c4 = c3 * c2
            c5 = c2 * d[..., 5] + c3 * d[..., 6] + c4 * d[..., 7] + 1
            c6 = c5 * c5
            c7 = 1.0 / c6
            c8 = a * d[..., 3]
            c9 = 2 * d[..., 2]
            c10 = 2 * c2
            c11 = 3 * c3
            c12 = c2 * d[..., 0]
            c13 = c3 * d[..., 1]
            c14 = c4 * d[..., 4]
            c15 = 2 * (c10 * d[..., 6] + c11 * d[..., 7] + d[..., 5]) * (c12 + c13 + c14 + 1)
            c16 = 2 * c10 * d[..., 1] + 2 * c11 * d[..., 4] + 2 * d[..., 0]
            c17 = 1 * c12 + 1 * c13 + 1 * c14 + 1
            c18 = b * d[..., 3]
            c19 = a * b
            c20 = -c15 * c19 + c16 * c19 * c15
            du_dx = c7 * (-c0 * c15 + c5 * (c0 * c16 + c17) + c6 * (b * c9 + 6.0 * c8))
            du_dy = c7 * (c20 + c6 * (a * c9 + 2 * c18))
            dv_dx = c7 * (c20 + c6 * (2 * a * d[2] + 2.0 * c18))
            dv_dy = c7 * (-c1 * c15 + c5 * (c1 * c16 + c17) + c6 * (6.0 * b * d[2] + 2.0 * c8))

    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError


class KannalaBrandtK3Transform:
    def distort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError

    def undistort(self, params: Tensor, points: Vector2) -> Vector2:
        raise NotImplementedError
