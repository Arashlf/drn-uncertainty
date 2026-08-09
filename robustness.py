from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drn import observable_certificate


class RobustnessRadiusError(ValueError):
    pass


def robustness_radius(hat_drn: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    hat_drn = np.asarray(hat_drn, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    if hat_drn.shape != q_hat.shape:
        raise RobustnessRadiusError(
            f"hat_drn and q_hat must share shape, got {hat_drn.shape} and {q_hat.shape}."
        )

    undefined = np.isnan(hat_drn)
    if np.any((~undefined) & ~np.isfinite(hat_drn)):
        raise RobustnessRadiusError("hat_drn must be finite wherever it is not NaN (unavailable).")
    if np.any((~undefined) & (hat_drn < 0)):
        raise RobustnessRadiusError("hat_drn must be nonnegative wherever defined.")

    zero_force = (~undefined) & (hat_drn == 0.0)
    exact_one = (~undefined) & (hat_drn == 1.0)
    growth = (~undefined) & (hat_drn > 1.0)
    decline = (~undefined) & (hat_drn < 1.0) & (~zero_force)

    used = (~undefined) & (~zero_force)
    if np.any(used & np.isnan(q_hat)):
        raise RobustnessRadiusError("q_hat is NaN where hat_drn is defined and nonzero.")
    if np.any(used & ((q_hat < 0) | (q_hat > 1))):
        raise RobustnessRadiusError("q_hat must lie in [0, 1] wherever hat_drn is defined and nonzero.")

    u_star = np.full(hat_drn.shape, np.nan)
    u_star = np.where(zero_force, 1.0, u_star)
    u_star = np.where(exact_one, 0.0, u_star)

    with np.errstate(divide="ignore", invalid="ignore"):
        g_den = hat_drn * q_hat
        g_val = np.where(g_den > 0, (hat_drn - 1.0) / g_den, 1.0)
        u_star = np.where(growth, np.minimum(1.0, g_val), u_star)

        d_den = (1.0 - hat_drn) + hat_drn * q_hat
        d_val = (1.0 - hat_drn) / d_den
        u_star = np.where(decline, np.minimum(1.0, d_val), u_star)

    return u_star


@dataclass
class RobustnessRadiusResult:
    u_star: np.ndarray
    hat_drn: np.ndarray
    q_hat: np.ndarray
    defined_mask: np.ndarray


def robustness_radius_from_observations(
    s: np.ndarray, x_tilde: np.ndarray, B: np.ndarray, gamma
) -> RobustnessRadiusResult:
    cert = observable_certificate(s, x_tilde, B, gamma, a=0.5)
    u_star = robustness_radius(cert.hat_drn, cert.q_hat)
    return RobustnessRadiusResult(
        u_star=u_star, hat_drn=cert.hat_drn, q_hat=cert.q_hat, defined_mask=cert.defined_mask
    )


@dataclass
class RadiusValidation:
    ok: bool
    reason: str


def validate_radius(u_star: np.ndarray, hat_drn: np.ndarray, q_hat: np.ndarray, primary_mask: np.ndarray) -> RadiusValidation:
    if not (u_star.shape == hat_drn.shape == q_hat.shape == primary_mask.shape):
        return RadiusValidation(
            False,
            f"shape mismatch: u_star={u_star.shape}, hat_drn={hat_drn.shape}, "
            f"q_hat={q_hat.shape}, primary_mask={primary_mask.shape}",
        )

    u_p = u_star[primary_mask]
    if not np.all(np.isfinite(u_p)):
        return RadiusValidation(False, "u_star contains a non-finite value on a primary-mask cell")
    if np.any((u_p < 0.0) | (u_p > 1.0)):
        return RadiusValidation(False, "u_star lies outside [0, 1] on a primary-mask cell")

    defined = ~np.isnan(hat_drn)
    exact_one = defined & (hat_drn == 1.0)
    if np.any(exact_one) and not np.all(u_star[exact_one] == 0.0):
        return RadiusValidation(False, "hat_drn == 1 does not give u_star == 0 exactly everywhere")

    zero_force = defined & (hat_drn == 0.0)
    if np.any(zero_force) and not np.all(u_star[zero_force] == 1.0):
        return RadiusValidation(False, "hat_drn == 0 does not give u_star == 1 exactly everywhere")

    return RadiusValidation(True, "ok")
