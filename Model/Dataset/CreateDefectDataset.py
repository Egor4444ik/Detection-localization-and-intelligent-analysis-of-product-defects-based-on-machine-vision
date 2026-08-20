import numpy as np
from .Augmentation import ObjectAugment

class DeformedObject(ObjectAugment):
    OBJECT = 0
    DENT = 1
    BUMP = 2
    CHIP = 3
    SCRATCH = 4
    LOCAL_DEFORMATION = 5

    def __init__(self, points):
        self.points = np.asarray(
            points,
            dtype=np.float64
        )
        super().__init__(points)
        self.center = None
        self.local_center = None
        self.normal = None
        self.tangent1 = None
        self.tangent2 = None
        self.u = None
        self.v = None
        self.w = None
        self.mask = None
        self.labels = np.zeros(len(self.points), dtype=np.int64)

    def _get_local_coordinate_system(self, radius):
        center = np.asarray(
            self.center,
            dtype=np.float64
        )
        distances = np.linalg.norm(
            self.points - center,
            axis=1
        )
        local_radius = radius
        for _ in range(5):
            indices = (
                distances <= local_radius
            )
            if np.sum(indices) >= 10:
                break
            local_radius *= 1.5
        local = self.points[indices]
        self.local_center = np.mean(
            local,
            axis=0
        )
        centered = (
            local - self.local_center
        )
        covariance = np.cov(
            centered.T
        )
        eigenvalues, eigenvectors = (
            np.linalg.eigh(covariance)
        )
        normal = eigenvectors[
            :,
            np.argmin(eigenvalues)
        ]
        self.normal = (
            normal
            /
            np.linalg.norm(normal)
        )
        tangent_candidates = (
            eigenvectors[
                :,
                np.argsort(eigenvalues)[-2:]
            ]
        )
        tangent1 = tangent_candidates[:, 0]
        tangent2 = tangent_candidates[:, 1]
        self.tangent1 = (
            tangent1
            /
            np.linalg.norm(tangent1)
        )
        self.tangent2 = (
            tangent2
            /
            np.linalg.norm(tangent2)
        )
        relative = (
            self.points - self.local_center
        )
        self.u = (
            relative @ self.tangent1
        )
        self.v = (
            relative @ self.tangent2
        )
        self.w = (
            relative @ self.normal
        )

    def _select_center(self):
        index = np.random.randint(
            0,
            len(self.points)
        )
        self.center = (
            self.points[index].copy()
        )

    def create_dent(self, radius, depth):
        self._select_center()
        self._get_local_coordinate_system(radius)
        distance = np.sqrt(
            self.u ** 2
            +
            self.v ** 2
        )
        mask = (
            distance <= radius
        )
        normalized_distance = (
            distance / radius
        )
        influence = np.zeros_like(
            distance
        )
        influence[mask] = (
            0.5
            *
            (
                1.0
                +
                np.cos(
                    np.pi
                    *
                    normalized_distance[mask]
                )
            )
        )
        new_points = (
            self.points.copy()
        )
        new_points[mask] -= (
            self.normal
            *
            depth
            *
            influence[mask, None]
        )
        self.points = new_points
        self.labels[mask] = (self.DENT)
        self.mask = mask

    def create_bump(self, radius, height):
        self._select_center()
        self._get_local_coordinate_system(radius)
        distance = np.sqrt(
            self.u ** 2
            +
            self.v ** 2
        )
        mask = (
            distance <= radius
        )
        normalized_distance = (
            distance / radius
        )
        influence = np.zeros_like(
            distance
        )
        influence[mask] = (
            0.5
            *
            (
                1.0
                +
                np.cos(
                    np.pi
                    *
                    normalized_distance[mask]
                )
            )
        )
        new_points = (
            self.points.copy()
        )
        new_points[mask] += (
            self.normal
            *
            height
            *
            influence[mask, None]
        )
        self.points = new_points
        self.labels[mask] = (
            self.BUMP
        )
        self.mask = mask

    def create_chip(self, radius, depth):
        self._select_center()
        self._get_local_coordinate_system(radius)
        distance = np.sqrt(
            self.u ** 2
            +
            self.v ** 2
        )
        mask = (
            distance <= radius
        )
        normalized_distance = (
            distance / radius
        )
        influence = np.zeros_like(
            distance
        )
        influence[mask] = (
            0.5
            *
            (
                1.0
                +
                np.cos(
                    np.pi
                    *
                    normalized_distance[mask]
                )
            )
        )
        new_points = (
            self.points.copy()
        )
        new_points[mask] -= (
            self.normal
            *
            depth
            *
            influence[mask, None]
        )
        self.points = new_points
        self.labels[mask] = (
            self.CHIP
        )
        self.mask = mask
    def create_scratch(self, length, width, depth):
        radius = (
            max(length, width)
            *
            1.2
        )
        self._select_center()
        self._get_local_coordinate_system(radius)
        mask = (
            (np.abs(self.u)
             <= length / 2.0)
            &
            (np.abs(self.v)
             <= width / 2.0)
        )
        new_points = (
            self.points.copy()
        )
        width_factor = (
            np.zeros_like(self.v)
        )
        width_factor[mask] = (
            1.0
            -
            np.abs(self.v[mask])
            /
            (width / 2.0)
        )
        length_factor = (
            np.zeros_like(self.u)
        )
        length_factor[mask] = (
            1.0
            -
            np.abs(self.u[mask])
            /
            (length / 2.0)
        )
        influence = (
            width_factor
            *
            length_factor
        )
        new_points[mask] -= (
            self.normal
            *
            depth
            *
            influence[mask, None]
        )
        self.points = new_points
        self.labels[mask] = (
            self.SCRATCH
        )
        self.mask = mask

    def create_local_deformation(self, radius, amplitude):
        self._select_center()
        self._get_local_coordinate_system(radius)
        distance_squared = (
            self.u ** 2
            +
            self.v ** 2
        )
        mask = (
            distance_squared
            <= radius ** 2
        )
        influence = np.exp(
            -distance_squared
            /
            (
                2.0
                *
                (radius / 2.0) ** 2
            )
        )
        influence[~mask] = 0.0
        new_points = (
            self.points.copy()
        )
        new_points += (
            self.normal
            *
            amplitude
            *
            influence[:, None]
        )
        self.points = new_points
        self.labels[mask] = (
            self.LOCAL_DEFORMATION
        )
        self.mask = mask

    def create_random_defects(self, min_defects=1, max_defects=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        defects_count = np.random.randint(
            min_defects,
            max_defects + 1
        )
        min_point = np.min(
            self.points,
            axis=0
        )
        max_point = np.max(
            self.points,
            axis=0
        )
        scene_size = np.linalg.norm(
            max_point - min_point
        )
        defect_types = [
            "dent",
            "bump",
            "chip",
            "scratch",
            "local_deformation"
        ]
        for _ in range(defects_count):
            defect_type = np.random.choice(
                defect_types
            )
            radius = np.random.uniform(
                scene_size * 0.05,
                scene_size * 0.15
            )
            avg_spacing = 2*scene_size / (len(self.points) ** (1/3))
            low_random_spacing, high_random_spacing = avg_spacing * 0.5, avg_spacing * 1
            
            if defect_type == "dent":
                radius = np.random.uniform(low_random_spacing, high_random_spacing)
                depth = np.random.uniform(low_random_spacing, high_random_spacing)                
                self.create_dent(
                    radius=radius,
                    depth=depth
                )
            elif defect_type == "bump":
                radius = np.random.uniform(low_random_spacing, high_random_spacing)
                height = np.random.uniform(low_random_spacing, high_random_spacing)
                self.create_bump(
                    radius=radius,
                    height=height
                )
            elif defect_type == "chip":
                radius = np.random.uniform(low_random_spacing, high_random_spacing)
                depth = np.random.uniform(low_random_spacing, high_random_spacing)
                self.create_chip(
                    radius=radius,
                    depth=depth
                )
            elif defect_type == "scratch":
                length = np.random.uniform(low_random_spacing, high_random_spacing)
                width = np.random.uniform(low_random_spacing, high_random_spacing)
                depth = np.random.uniform(low_random_spacing, high_random_spacing)
                self.create_scratch(
                    length=length,
                    width=width,
                    depth=depth
                )
            elif defect_type == "local_deformation":
                radius = np.random.uniform(low_random_spacing, high_random_spacing)
                amplitude = np.random.uniform(low_random_spacing, high_random_spacing)
                self.create_local_deformation(
                    radius=radius,
                    amplitude=amplitude
                )

        self.full_augment()

        return self.points, self.labels