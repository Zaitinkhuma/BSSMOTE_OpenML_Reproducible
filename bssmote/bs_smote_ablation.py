
import os
import random
import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from imblearn.base import BaseSampler
from imblearn.over_sampling import ADASYN
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


def set_reproducible_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


class BSSMOTE(BaseSampler):
    _sampling_type = "over-sampling"
    _parameter_constraints = {}

    def __init__(
        self,
        n_clusters="auto",
        adasyn_neighbors="auto",
        svm_margin=0.75,
        noise_scale=0.05,
        cluster_radius=1.25,
        svm_kernel="linear",
        svm_C=1.0,
        svm_gamma="scale",
        random_state=42
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.adasyn_neighbors = adasyn_neighbors
        self.svm_margin = svm_margin
        self.noise_scale = noise_scale
        self.cluster_radius = cluster_radius
        self.svm_kernel = svm_kernel
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.random_state = 42 if random_state is None else random_state
        self.X_noise_ = None
        self.ablation_name_ = "Full BSSMOTE"

    def _seed(self):
        set_reproducible_seed(self.random_state)

    def _validate_binary_input(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return False, classes, counts
        if counts.min() < 2:
            return False, classes, counts
        return True, classes, counts

    def _safe_neighbors(self, X_min):
        n = len(X_min)
        if n <= 1:
            return 1
        if self.adasyn_neighbors == "auto":
            return max(1, min(5, n - 1))
        return max(1, min(int(self.adasyn_neighbors), n - 1))

    def _safe_n_clusters(self, X_maj):
        n = len(X_maj)
        if n <= 1:
            return 1
        if self.n_clusters == "auto":
            return max(1, min(10, max(1, n // 20)))
        return max(1, min(int(self.n_clusters), n))

    def _majority_cleaning(self, X_min, X_maj, minority, majority):
        self._seed()

        if len(X_maj) <= 1:
            X_clean = np.vstack([X_min, X_maj])
            y_clean = np.hstack([
                np.full(len(X_min), minority),
                np.full(len(X_maj), majority)
            ])
            return X_clean, y_clean

        n_clust = self._safe_n_clusters(X_maj)

        kmeans = KMeans(
            n_clusters=n_clust,
            random_state=self.random_state,
            n_init=10,
            algorithm="lloyd"
        )

        labels = kmeans.fit_predict(X_maj)
        centers = kmeans.cluster_centers_

        dists = np.linalg.norm(X_maj - centers[labels], axis=1)
        mean_dist = dists.mean() + 1e-12
        keep_mask = dists <= self.cluster_radius * mean_dist

        X_maj_clean = X_maj[keep_mask]
        y_maj_clean = np.full(len(X_maj_clean), majority)

        X_clean = np.vstack([X_min, X_maj_clean])
        y_clean = np.hstack([
            np.full(len(X_min), minority),
            y_maj_clean
        ])

        return X_clean, y_clean

    def _get_new_minority_samples(self, X_before, y_before, X_after, y_after, minority):
        n_old = int(np.sum(y_before == minority))
        n_new = int(np.sum(y_after == minority) - n_old)
        if n_new <= 0:
            return None
        return X_after[y_after == minority][-n_new:]

    def _fit_boundary_model(self, X_train, y_train):
        self._seed()

        if len(np.unique(y_train)) < 2:
            raise ValueError("Boundary model needs two classes.")

        if self.svm_kernel == "linear":
            model = SGDClassifier(
                loss="hinge",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-4,
                shuffle=False,
                random_state=self.random_state
            )
        elif self.svm_kernel == "rbf":
            model = SVC(
                kernel="rbf",
                C=self.svm_C,
                gamma=self.svm_gamma,
                random_state=self.random_state
            )
        else:
            raise ValueError("svm_kernel must be 'linear' or 'rbf'.")

        model.fit(X_train, y_train)
        return model

    def _boundary_refinement(self, X_train, y_train, X_candidates, minority):
        self._seed()
        rng = np.random.RandomState(self.random_state)

        # Reset lightweight diagnostics for each call. These attributes do not
        # change the external API, but they make zero-generation cases traceable.
        self.boundary_candidates_seen_ = 0
        self.boundary_candidates_accepted_ = 0
        self.boundary_fallback_used_ = False

        if X_candidates is None or len(X_candidates) == 0:
            return None

        X_candidates = np.asarray(X_candidates, dtype=float)
        self.boundary_candidates_seen_ = int(len(X_candidates))

        try:
            svm = self._fit_boundary_model(X_train, y_train)
            decision = svm.decision_function(X_candidates)
        except Exception:
            return None

        if np.ndim(decision) > 1:
            decision = decision[:, 0]

        decision = np.asarray(decision, dtype=float)
        finite_mask = np.isfinite(decision)

        if not np.any(finite_mask):
            return None

        X_candidates = X_candidates[finite_mask]
        decision = decision[finite_mask]
        decision_abs = np.abs(decision)

        if len(X_candidates) == 0:
            return None

        # The previous fixed threshold, svm_margin * 0.25, was too strict for
        # raw SGD/SVC decision scores and could reject all candidates, especially
        # for Full BSSMOTE Linear. Keep the same boundary-sensitive concept, but
        # add a scale-aware fallback: if the fixed threshold accepts none, retain
        # the closest candidates to the learned boundary.
        fixed_threshold = float(self.svm_margin) * 0.25
        boundary_mask = decision_abs <= fixed_threshold

        if not np.any(boundary_mask):
            keep_fraction = 0.10 if self.svm_kernel == "linear" else 0.05
            keep_n = max(1, int(np.ceil(keep_fraction * len(X_candidates))))
            keep_n = min(keep_n, len(X_candidates))

            closest_idx = np.argsort(decision_abs)[:keep_n]
            boundary_mask = np.zeros(len(X_candidates), dtype=bool)
            boundary_mask[closest_idx] = True
            self.boundary_fallback_used_ = True

        X_boundary = X_candidates[boundary_mask]
        boundary_dist = decision_abs[boundary_mask]
        signed_decision = decision[boundary_mask]

        if len(X_boundary) == 0:
            return None

        self.boundary_candidates_accepted_ = int(len(X_boundary))

        # Noise should be strongest very near the boundary and weaker away from
        # it. Use a clipped, data-scale-aware denominator so the scale never
        # becomes negative when raw decision scores exceed svm_margin.
        denom = max(float(np.nanmax(boundary_dist)), float(self.svm_margin), 1e-12)
        scale = 1.0 - (boundary_dist / (denom + 1e-12))
        scale = np.clip(scale, 0.05, 1.0)[:, None]

        gaussian = rng.normal(loc=0.0, scale=self.noise_scale, size=X_boundary.shape)

        if self.svm_kernel == "linear" and hasattr(svm, "coef_"):
            w = svm.coef_.reshape(-1)
            w_norm = np.linalg.norm(w) + 1e-12
            direction = w / w_norm

            # Convert the raw decision score to an approximate signed distance.
            # This avoids over-large perturbations from uncalibrated SGD scores.
            signed_distance = signed_decision / w_norm
            pull_to_boundary = -signed_distance[:, None] * direction

            X_noisy = (
                X_boundary
                + scale * (pull_to_boundary * 0.5)
                + gaussian * scale * 0.02
            )
        else:
            X_noisy = X_boundary + gaussian * scale * 0.02

        return X_noisy

    def _adasyn_resample(self, X_clean, y_clean, X_min):
        self._seed()

        if len(np.unique(y_clean)) < 2:
            return X_clean, y_clean

        k = self._safe_neighbors(X_min)
        if k < 1:
            return X_clean, y_clean

        ada = ADASYN(n_neighbors=k, random_state=self.random_state)
        return ada.fit_resample(X_clean, y_clean)

    def _fit_resample(self, X, y):
        self._seed()

        X, y, *_ = self._check_X_y(X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        is_valid, classes, counts = self._validate_binary_input(X, y)
        if not is_valid:
            return X, y

        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]

        X_min = X[y == minority]
        X_maj = X[y == majority]

        X_clean, y_clean = self._majority_cleaning(X_min, X_maj, minority, majority)

        if len(np.unique(y_clean)) < 2:
            return X, y

        try:
            X_ada, y_ada = self._adasyn_resample(X_clean, y_clean, X_min)
        except Exception:
            return X_clean, y_clean

        X_synth = self._get_new_minority_samples(X_clean, y_clean, X_ada, y_ada, minority)
        if X_synth is None:
            return X_clean, y_clean

        X_noisy = self._boundary_refinement(X_clean, y_clean, X_synth, minority)
        if X_noisy is None:
            return X_clean, y_clean

        y_noisy = np.full(len(X_noisy), minority)
        self.X_noise_ = X_noisy

        return np.vstack([X_clean, X_noisy]), np.hstack([y_clean, y_noisy])


class BSSMOTE_Full(BSSMOTE):
    pass


class BSSMOTE_NoClustering(BSSMOTE):
    def _fit_resample(self, X, y):
        self._seed()

        X, y, *_ = self._check_X_y(X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        is_valid, classes, counts = self._validate_binary_input(X, y)
        if not is_valid:
            return X, y

        minority = classes[np.argmin(counts)]
        X_min = X[y == minority]
        X_clean = X.copy()
        y_clean = y.copy()

        try:
            X_ada, y_ada = self._adasyn_resample(X_clean, y_clean, X_min)
        except Exception:
            return X_clean, y_clean

        X_synth = self._get_new_minority_samples(X_clean, y_clean, X_ada, y_ada, minority)
        if X_synth is None:
            return X_clean, y_clean

        X_noisy = self._boundary_refinement(X_clean, y_clean, X_synth, minority)
        if X_noisy is None:
            return X_clean, y_clean

        y_noisy = np.full(len(X_noisy), minority)
        self.X_noise_ = X_noisy

        return np.vstack([X_clean, X_noisy]), np.hstack([y_clean, y_noisy])


class BSSMOTE_NoADASYN(BSSMOTE):
    def _fit_resample(self, X, y):
        self._seed()

        X, y, *_ = self._check_X_y(X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        is_valid, classes, counts = self._validate_binary_input(X, y)
        if not is_valid:
            return X, y

        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]

        X_min = X[y == minority]
        X_maj = X[y == majority]

        X_clean, y_clean = self._majority_cleaning(X_min, X_maj, minority, majority)

        if len(np.unique(y_clean)) < 2:
            return X, y

        X_noisy = self._boundary_refinement(X_clean, y_clean, X_min.copy(), minority)
        if X_noisy is None:
            return X_clean, y_clean

        y_noisy = np.full(len(X_noisy), minority)
        self.X_noise_ = X_noisy

        return np.vstack([X_clean, X_noisy]), np.hstack([y_clean, y_noisy])


class BSSMOTE_NoSVM(BSSMOTE):
    def _fit_resample(self, X, y):
        self._seed()

        X, y, *_ = self._check_X_y(X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        is_valid, classes, counts = self._validate_binary_input(X, y)
        if not is_valid:
            return X, y

        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]

        X_min = X[y == minority]
        X_maj = X[y == majority]

        X_clean, y_clean = self._majority_cleaning(X_min, X_maj, minority, majority)

        try:
            return self._adasyn_resample(X_clean, y_clean, X_min)
        except Exception:
            return X_clean, y_clean
