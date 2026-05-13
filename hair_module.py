from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from bm3d import bm3d
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


def interpolate_emmisivity(e_pixel, good_band_indices, all_band_indices):
    e_interp = jnp.interp(all_band_indices, good_band_indices, e_pixel)
    return e_interp


@jax.jit
def interpolate_emmisivity_vmap(e_sparse, good_band_indices, all_band_indices):
    e_full = jax.vmap(interpolate_emmisivity, in_axes=(0, None, None))(
        e_sparse, good_band_indices, all_band_indices
    )
    return e_full


def interpolate_sky(S_sky, good_indices, all_indices):
    S_sky_full = UnivariateSpline(good_indices, S_sky, k=3, s=0)(all_indices)
    return S_sky_full


def detect_noisy_bands(
    hsi_data,
    row_mean_threshold=1.05,
    band_problem_ratio=0.05,
    max_bad_band_ratio=0.3,
    smooth_sigma=2.0,
    plot=True,
):
    H, W, C = hsi_data.shape
    spatial_means = np.mean(hsi_data, axis=1)
    spatial_len = H
    line_label = "row"
    spatial_means_smooth = gaussian_filter1d(
        spatial_means, sigma=smooth_sigma, axis=0, mode="nearest"
    )
    residual = spatial_means - spatial_means_smooth
    stripe_strengths = np.sqrt(np.mean(residual**2, axis=0))
    prev_lines = spatial_means[:-2, :]
    curr_lines = spatial_means[1:-1, :]
    next_lines = spatial_means[2:, :]
    higher_mask = (curr_lines > row_mean_threshold * prev_lines) & (
        curr_lines > row_mean_threshold * next_lines
    )
    lower_mask = (curr_lines * row_mean_threshold < prev_lines) & (
        curr_lines * row_mean_threshold < next_lines
    )
    problem_mask = higher_mask | lower_mask
    problem_line_count = np.sum(problem_mask, axis=0)
    noisy_mask = (problem_line_count / spatial_len) >= band_problem_ratio
    noisy_band_indices = np.where(noisy_mask)[0].tolist()
    noisy_band_ratio = len(noisy_band_indices) / C
    if noisy_band_ratio > max_bad_band_ratio:
        max_bad_bands = int(max_bad_band_ratio * C)
        top_indices = np.argsort(stripe_strengths)[-max_bad_bands:]
        noisy_band_indices = sorted(top_indices.tolist())
        print(
            f"Warning: detected noisy band ratio = {noisy_band_ratio:.3f}, "
            f"which exceeds the maximum allowed ratio = {max_bad_band_ratio:.3f}. "
            f"Only the top {max_bad_bands} bands with strongest noise are kept."
        )
    else:
        print(f"Detection completed: {len(noisy_band_indices)} noisy bands found.")
    if plot:
        band_indices = np.arange(C)
        noisy_set = set(noisy_band_indices)
        colors = ["C1" if b in noisy_set else "C0" for b in band_indices]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.bar(band_indices, stripe_strengths, color=colors, width=0.8, alpha=0.8)
        ax1.set_ylabel("Stripe noise strength")
        ax1.set_title("Noise score per band (stripe strength)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(
            handles=[
                Patch(facecolor="C0", alpha=0.8, label="Normal band"),
                Patch(facecolor="C1", alpha=0.8, label="Noisy band"),
            ],
            loc="upper right",
        )
        ax2.bar(band_indices, problem_line_count, color=colors, width=0.8, alpha=0.8)
        threshold_lines = int(np.ceil(band_problem_ratio * spatial_len))
        ax2.axhline(
            y=threshold_lines,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({threshold_lines} {line_label}s)",
        )
        ax2.set_xlabel("Band index")
        ax2.set_ylabel(f"Problem {line_label} count")
        ax2.set_title(f"Noisy {line_label} count per band")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    return noisy_band_indices, stripe_strengths


def lookup_B_vector_batch(T, B_lut, config):
    t_norm = jax.numpy.clip((T - config[0]) / (config[1] - config[0]), 0, 1)
    num_T_samples = B_lut.shape[0]
    idx_float = t_norm * (num_T_samples - 1)
    idx_low = jax.numpy.floor(idx_float).astype(jax.numpy.int32)
    idx_high = jax.numpy.ceil(idx_float).astype(jax.numpy.int32)
    weight = idx_float - idx_low
    weight = weight[:, None]
    idx_low = jax.numpy.clip(idx_low, 0, num_T_samples - 1)
    idx_high = jax.numpy.clip(idx_high, 0, num_T_samples - 1)
    B_vector = (1 - weight) * B_lut[idx_low, :] + weight * B_lut[idx_high, :]
    return B_vector


def soft_threshold(x, t):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - t, 0.0)


def precompute_kernels(H, W, params):
    dx = jnp.zeros((H, W)).at[0, 0].set(1).at[0, 1].set(-1)
    dy = jnp.zeros((H, W)).at[0, 0].set(1).at[1, 0].set(-1)
    dyy = jnp.zeros((H, W)).at[0, 0].set(1).at[1, 0].set(-2).at[2, 0].set(1)
    dx_f = jnp.fft.rfft2(dx)
    dy_f = jnp.fft.rfft2(dy)
    dyy_f = jnp.fft.rfft2(dyy)
    dxt_f = jnp.conj(dx_f)
    dyt_f = jnp.conj(dy_f)
    dyyt_f = jnp.conj(dyy_f)
    rho1 = params[6]
    rho2 = params[7]
    rho3 = params[8]
    rho4 = params[9]
    Denom_X = 1.0 / (
        1.0
        + rho1 * (jnp.abs(dx_f) ** 2 + jnp.abs(dy_f) ** 2)
        + rho2 * (jnp.abs(dyy_f) ** 2)
    )
    Denom_S = 1.0 / (1.0 + rho3 * jnp.abs(dx_f) ** 2 + rho4)
    return dx_f, dy_f, dyy_f, dxt_f, dyt_f, dyyt_f, Denom_X, Denom_S


def admm_step(state, _iter_idx, constants):
    X, S, Dx, Dy, Hyy, Ex, F, U_dx, U_dy, U_hyy, U_ex, U_f = state
    Y_f, kernels, denoms, p, image_shape = constants
    dx_f, dy_f, dyy_f, dxt_f, dyt_f, dyyt_f = kernels
    Denom_X, Denom_S = denoms
    lambda_TVx = p[0]
    lambda_TVy = p[1]
    lambda_HOTV = p[2]
    lambda1 = p[3]
    lambda2 = p[4]
    rho_tv1 = p[6]
    rho_tv2 = p[7]
    rho_utv = p[8]
    rho_s = p[9]
    S_f = jnp.fft.rfft2(S)
    V_dx = Dx + U_dx
    V_dy = Dy + U_dy
    V_hyy = Hyy + U_hyy
    RHS_X_f = (
        Y_f
        - S_f
        + rho_tv1 * (dxt_f * jnp.fft.rfft2(V_dx))
        + rho_tv1 * (dyt_f * jnp.fft.rfft2(V_dy))
        + rho_tv2 * (dyyt_f * jnp.fft.rfft2(V_hyy))
    )
    X_f = RHS_X_f * Denom_X
    X = jnp.fft.irfft2(X_f, s=image_shape)
    V_ex = Ex + U_ex
    V_f = F + U_f
    RHS_S_f = (
        Y_f - X_f + rho_utv * (dxt_f * jnp.fft.rfft2(V_ex)) + rho_s * jnp.fft.rfft2(V_f)
    )
    S_f = RHS_S_f * Denom_S
    S = jnp.fft.irfft2(S_f, s=image_shape)
    Grad_X_x = jnp.fft.irfft2(X_f * dx_f, s=image_shape)
    Grad_X_y = jnp.fft.irfft2(X_f * dy_f, s=image_shape)
    Grad_X_yy = jnp.fft.irfft2(X_f * dyy_f, s=image_shape)
    Grad_S_x = jnp.fft.irfft2(S_f * dx_f, s=image_shape)
    Dx_new = soft_threshold(Grad_X_x - U_dx, lambda_TVx / rho_tv1)
    Dy_new = soft_threshold(Grad_X_y - U_dy, lambda_TVy / rho_tv1)
    Hyy_new = soft_threshold(Grad_X_yy - U_hyy, lambda_HOTV / rho_tv2)
    Ex_new = soft_threshold(Grad_S_x - U_ex, lambda1 / rho_utv)
    F_new = soft_threshold(S - U_f, lambda2 / rho_s)
    U_dx_new = U_dx - (Grad_X_x - Dx_new)
    U_dy_new = U_dy - (Grad_X_y - Dy_new)
    U_hyy_new = U_hyy - (Grad_X_yy - Hyy_new)
    U_ex_new = U_ex - (Grad_S_x - Ex_new)
    U_f_new = U_f - (S - F_new)
    new_state = (
        X,
        S,
        Dx_new,
        Dy_new,
        Hyy_new,
        Ex_new,
        F_new,
        U_dx_new,
        U_dy_new,
        U_hyy_new,
        U_ex_new,
        U_f_new,
    )
    return new_state, None


def destripe_band(Y_k, strength, kernels, denoms, params):
    H, W = Y_k.shape
    Y_f = jnp.fft.rfft2(Y_k)
    params_band = [
        params[0] * strength,
        params[1] * strength,
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
    ]
    X = Y_k
    S = jnp.zeros_like(Y_k)
    zeros = jnp.zeros_like(Y_k)
    init_state = (
        X,
        S,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
    )
    image_shape = (H, W)
    constants = (Y_f, kernels, denoms, params_band, image_shape)
    final_state, _ = jax.lax.scan(
        partial(admm_step, constants=constants),
        init_state,
        None,
        length=params_band[5],
    )
    X_final, S_final = final_state[0], final_state[1]
    return X_final, S_final


@jax.jit
def destripe(
    stripe_strengths,
    hsi,
    lambda_TVx=0.004,
    lambda_TVy=0.002,
    lambda_HOTV=0.005,
    lambda1=1.0,
    lambda2=0.005,
    max_iter=100,
    rho_tv1=0.1,
    rho_tv2=0.05,
    rho_utv=0.2,
    rho_s=0.05,
):
    params = [
        lambda_TVx,
        lambda_TVy,
        lambda_HOTV,
        lambda1,
        lambda2,
        max_iter,
        rho_tv1,
        rho_tv2,
        rho_utv,
        rho_s,
    ]
    H, W, _ = hsi.shape
    kernels_and_denoms = precompute_kernels(H, W, params)
    kernels = kernels_and_denoms[:6]
    denoms = kernels_and_denoms[6:]
    batch_core = jax.vmap(
        destripe_band, in_axes=(2, 0, None, None, None), out_axes=(2, 2)
    )
    denoised_hsi, noise = batch_core(hsi, stripe_strengths, kernels, denoms, params)
    return noise, denoised_hsi


def est_noise(y):
    small = 1e-6
    L, N = y.shape
    RR = y @ y.T
    RRi = np.linalg.inv(RR + small * np.eye(L))
    denom = np.diag(RRi)[:, np.newaxis]
    w = (RRi @ y) / denom
    Rw = np.diag(np.sum(w**2, axis=1) / N)
    return w, Rw


def hysime(
    y,
    w,
    Rw,
):
    L, N = y.shape
    x = y - w
    Ry = (y @ y.T) / N
    Rx = (x @ x.T) / N
    E, _, _ = np.linalg.svd(Rx, full_matrices=True)
    mu = np.sum(np.diag(Rx)) / L / 1e5
    Rn = Rw + mu * np.eye(L)
    Py = np.diag(E.T @ Ry @ E)
    Pn = np.diag(E.T @ Rn @ E)
    cost_F = -Py + 2.0 * Pn
    kf = int(np.sum(cost_F < 0))
    print(f"The estimated signal subspace dimension is: k = {kf:d}")
    return kf, E


def _denoise_single_band(eigen_im_row, base_sigma, H, W):
    min_x = eigen_im_row.min()
    max_x = eigen_im_row.max()
    eigen_im_row_shifted = eigen_im_row - min_x
    scale = max_x - min_x
    if scale == 0:
        scale = 1.0
    eigen_im_2d = eigen_im_row_shifted.reshape(H, W) / scale
    sigma = base_sigma / scale
    filt_eigen_im = bm3d(eigen_im_2d, sigma_psd=float(sigma))
    return (filt_eigen_im * scale + min_x).reshape(-1)


def denoise(img, k_subspace=None, n_jobs=-1):
    img = np.asarray(img, dtype=np.float64)
    H, W, C = img.shape
    N = H * W
    Y = img.reshape(N, C).T
    _, Rw0 = est_noise(Y)
    dRw0 = np.diag(Rw0)
    scale_factor = 1.0 / np.sqrt(dRw0 + 1e-12)
    Y = scale_factor[:, None] * Y
    w, Rw = est_noise(Y)
    if k_subspace is None:
        k_subspace, E = hysime(Y, w, Rw)
    else:
        _, E = hysime(Y, w, Rw)
    E = E[:, :k_subspace]
    eigen_Y = E.T @ Y
    dRw = np.diag(Rw)
    E2 = E**2
    base_sigmas = np.sqrt(dRw @ E2)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_denoise_single_band)(
            eigen_Y[i, :],
            base_sigmas[i],
            H,
            W,
        )
        for i in range(k_subspace)
    )
    eigen_Y_bm3d = np.array(results)
    Y_reconst = E @ eigen_Y_bm3d
    Y_reconst = np.sqrt(dRw0)[:, None] * Y_reconst
    image_fasthyde = Y_reconst.T.reshape(H, W, C)
    return image_fasthyde


_FWHM_TO_SIGMA = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0))


@jax.jit
def trapz(y, x):
    dx = jnp.diff(x)
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def z_norm(x):
    x = np.asarray(x, dtype=np.float32)
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1e-12
    x_z = (x - x_mean) / x_std
    return x_z, x_mean, x_std


@jax.jit
def forward_model_constant_srf(params, bands_idx, wl_ref, signal_ref_z):
    c0, c1, c2, w = params

    sigma = jnp.maximum(w / _FWHM_TO_SIGMA, 1e-6)
    centers = c0 + c1 * bands_idx + c2 * (bands_idx**2)

    wl_diff = (wl_ref[None, :] - centers[:, None]) / sigma
    srf = jnp.exp(-0.5 * (wl_diff**2))
    area = trapz(srf, wl_ref)
    srf_norm = srf / jnp.maximum(area[:, None], 1e-12)

    signal_sim = trapz(signal_ref_z[None, :] * srf_norm, wl_ref)
    return signal_sim


@jax.jit
def correct_band_score(params, bands_idx, wl_ref, signal_ref_z, signal_measured_z):
    signal_sim = forward_model_constant_srf(params, bands_idx, wl_ref, signal_ref_z)
    residual = signal_sim - signal_measured_z
    return jnp.sum(residual**2)


batch_correct_band_score = jax.jit(
    jax.vmap(
        correct_band_score,
        in_axes=(0, None, None, None, None),
    )
)


def correct_band(sky, wav, path_to_ref_sky, show_plot=True):
    sky = np.asarray(sky, dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float32)

    wl_ref_highres, sky_ref_highres = np.loadtxt(
        path_to_ref_sky, usecols=(0, 1), unpack=True
    )
    wl_ref_highres = np.asarray(wl_ref_highres, dtype=np.float32)
    sky_ref_highres = np.asarray(sky_ref_highres, dtype=np.float32)

    wl_ref_highres = wl_ref_highres / 1000.0

    wav_min = float(np.min(wav)) - 0.02
    wav_max = float(np.max(wav)) + 0.02
    overlap_mask = (wl_ref_highres >= wav_min) & (wl_ref_highres <= wav_max)

    wl_ref_highres = wl_ref_highres[overlap_mask]
    sky_ref_highres = sky_ref_highres[overlap_mask]

    sky_ref_z, ref_mean, ref_std = z_norm(sky_ref_highres)
    sky_z, _, _ = z_norm(sky)

    n_bands = wav.size
    bands_idx = np.arange(n_bands, dtype=np.float32)

    c0_guess = float(wav[0])
    c1_guess = float((wav[-1] - wav[0]) / (n_bands - 1))

    lower_bounds = np.array(
        [c0_guess - 0.08, c1_guess * 0.9, -1e-6, 0.01],
        dtype=np.float32,
    )
    upper_bounds = np.array(
        [c0_guess + 0.08, c1_guess * 1.1, 1e-6, 0.05],
        dtype=np.float32,
    )

    global_lower_bounds = lower_bounds.copy()
    global_upper_bounds = upper_bounds.copy()

    best_params = None
    best_score = np.inf

    bands_idx_jax = jnp.asarray(bands_idx)
    wl_ref_jax = jnp.asarray(wl_ref_highres)
    sky_ref_z_jax = jnp.asarray(sky_ref_z)
    sky_z_jax = jnp.asarray(sky_z)

    grid_levels = [10, 10, 10]

    for level, n_points in enumerate(grid_levels):
        c0_grid = np.linspace(
            lower_bounds[0], upper_bounds[0], n_points, dtype=np.float32
        )
        c1_grid = np.linspace(
            lower_bounds[1], upper_bounds[1], n_points, dtype=np.float32
        )
        c2_grid = np.linspace(
            lower_bounds[2], upper_bounds[2], n_points, dtype=np.float32
        )
        w_grid = np.linspace(
            lower_bounds[3], upper_bounds[3], n_points, dtype=np.float32
        )

        C0, C1, C2, W = np.meshgrid(c0_grid, c1_grid, c2_grid, w_grid, indexing="ij")
        params_grid = np.stack((C0, C1, C2, W), axis=-1).reshape(-1, 4)

        scores = batch_correct_band_score(
            jnp.asarray(params_grid),
            bands_idx_jax,
            wl_ref_jax,
            sky_ref_z_jax,
            sky_z_jax,
        )
        scores_np = np.asarray(scores)
        finite_mask = np.isfinite(scores_np)
        if not np.any(finite_mask):
            continue

        level_best_idx = int(np.argmin(np.where(finite_mask, scores_np, np.inf)))
        level_best_score = float(scores_np[level_best_idx])
        level_best_params = params_grid[level_best_idx]

        if level_best_score < best_score:
            best_score = level_best_score
            best_params = level_best_params.copy()

        if level < len(grid_levels) - 1:
            span = (upper_bounds - lower_bounds) / (n_points - 1)
            lower_bounds = np.maximum(level_best_params - span, global_lower_bounds)
            upper_bounds = np.minimum(level_best_params + span, global_upper_bounds)

    if best_params is None:
        raise RuntimeError(
            "Grid search failed: all candidate scores are NaN/Inf. "
            "Please check input sky/reference data quality and wavelength overlap."
        )

    c0_opt, c1_opt, c2_opt, w_opt = best_params
    cor_wav = c0_opt + c1_opt * bands_idx + c2_opt * (bands_idx**2)

    simulated_sky_z = np.asarray(
        forward_model_constant_srf(
            jnp.asarray(best_params),
            bands_idx_jax,
            wl_ref_jax,
            sky_ref_z_jax,
        )
    )
    simulated_sky = simulated_sky_z * ref_std + ref_mean

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(
            cor_wav,
            simulated_sky_z,
            "b--",
            linewidth=2,
            label="Convolved reference (z-score)",
        )
        plt.plot(
            cor_wav,
            sky_z,
            "g.-",
            linewidth=1.5,
            markersize=4,
            label="Measured sky (z-score)",
        )
        plt.xlabel(r"Wavelength ($\mu m$)")
        plt.ylabel("Z-score")
        plt.title("Reference convolved vs. measured sky")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return cor_wav, simulated_sky, float(w_opt)
