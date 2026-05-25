#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class ProfileFitResult:
    position_cm: float
    file_name: str
    axis: str
    e2_radius_um: float
    fwhm_um: float
    second_moment_width_um: float
    r_squared: float


@dataclass
class M2FitResult:
    m2: float
    w0_um: float
    z0_cm: float
    m2_uncertainty: float
    w0_uncertainty_um: float
    r_squared: float


def gaussian(x: np.ndarray, amp: float, mean: float, sigma: float, c: float) -> np.ndarray:
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + c


def extract_position(filename: str, base_filename: str, file_extension: str) -> float:
    match = re.search(rf"{re.escape(base_filename)}([+-]?\d+)cm_raw{re.escape(file_extension)}$", filename)
    if match:
        return float(int(match.group(1)))
    if filename == f"{base_filename}0cm_raw{file_extension}":
        return 0.0
    return float("inf")


def list_sorted_files(folder_path: Path, base_filename: str, file_extension: str) -> List[Tuple[float, Path]]:
    all_files = [p for p in folder_path.iterdir() if p.is_file() and p.name.endswith(file_extension)]
    positioned = []
    for p in all_files:
        pos = extract_position(p.name, base_filename, file_extension)
        if np.isfinite(pos):
            positioned.append((pos, p))
    positioned.sort(key=lambda item: item[0])
    return positioned


def fit_profile(profile: np.ndarray, pixel_pitch_um: float) -> Dict[str, float]:
    x = np.arange(len(profile), dtype=float)
    y = profile.astype(float)

    p0 = [float(np.max(y)), float(np.argmax(y)), max(len(y) / 10, 1.0), float(np.min(y))]
    bounds = ([0.0, 0.0, 1e-6, -np.inf], [np.inf, len(y), np.inf, np.inf])
    popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=20000)

    y_fit = gaussian(x, *popt)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)

    sigma_px = abs(float(popt[2]))
    e2_radius_um = np.sqrt(2.0) * sigma_px * pixel_pitch_um
    fwhm_um = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_px * pixel_pitch_um

    total = float(np.sum(y))
    if total <= 0:
        second_moment_width_um = float("nan")
    else:
        centroid = float(np.sum(x * y) / total)
        variance = float(np.sum(((x - centroid) ** 2) * y) / total)
        sigma_second_moment_px = np.sqrt(max(variance, 0.0))
        second_moment_width_um = 4.0 * sigma_second_moment_px * pixel_pitch_um  # D4σ

    return {
        "e2_radius_um": float(e2_radius_um),
        "fwhm_um": float(fwhm_um),
        "second_moment_width_um": float(second_moment_width_um),
        "r_squared": float(r_squared),
        "x": x,
        "y": y,
        "y_fit": y_fit,
    }


def calculate_m2(positions_cm: np.ndarray, beam_radii_um: np.ndarray, wavelength_um: float) -> M2FitResult:
    valid_mask = ~np.isnan(positions_cm) & ~np.isnan(beam_radii_um)
    if np.sum(valid_mask) < 3:
        raise ValueError("Need at least 3 valid points for M² calculation")

    z_m = positions_cm[valid_mask] * 0.01
    w_squared_m2 = (beam_radii_um[valid_mask] * 1e-6) ** 2

    min_idx = int(np.argmin(beam_radii_um[valid_mask]))
    z0_m = z_m[min_idx]
    z_shifted_m = z_m - z0_m

    coeffs, covariance = np.polyfit(z_shifted_m**2, w_squared_m2, 1, cov=True)
    b_coeff = float(coeffs[0])
    w0_squared = float(coeffs[1])

    if w0_squared <= 0 or b_coeff <= 0:
        raise ValueError(f"Invalid fit parameters: w0²={w0_squared}, b={b_coeff}")

    b_unc = float(np.sqrt(covariance[0, 0]))
    w0_squared_unc = float(np.sqrt(covariance[1, 1]))

    w0_m = np.sqrt(w0_squared)
    w0_unc_m = 0.5 * w0_squared_unc / w0_m

    m2 = np.pi * w0_m * np.sqrt(b_coeff) / (wavelength_um * 1e-6)

    dM2_db = np.pi * w0_m / (2.0 * np.sqrt(b_coeff) * wavelength_um * 1e-6)
    dM2_dw0 = np.pi * np.sqrt(b_coeff) / (wavelength_um * 1e-6)
    m2_unc = np.sqrt((dM2_db * b_unc) ** 2 + (dM2_dw0 * w0_unc_m) ** 2)

    w_pred = np.sqrt(w0_squared + b_coeff * z_shifted_m**2)
    ss_res = float(np.sum((np.sqrt(w_squared_m2) - w_pred) ** 2))
    ss_tot = float(np.sum((np.sqrt(w_squared_m2) - np.mean(np.sqrt(w_squared_m2))) ** 2))
    fit_r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)

    return M2FitResult(
        m2=float(m2),
        w0_um=float(w0_m * 1e6),
        z0_cm=float(z0_m * 100.0),
        m2_uncertainty=float(m2_unc),
        w0_uncertainty_um=float(w0_unc_m * 1e6),
        r_squared=float(fit_r2),
    )


def calculate_rayleigh_range_cm(m2: float, w0_um: float, wavelength_um: float) -> float:
    w0_m = w0_um * 1e-6
    lambda_m = wavelength_um * 1e-6
    return float(np.pi * w0_m**2 / (m2 * lambda_m) * 100.0)


def calculate_divergence_half_angle_mrad(m2: float, w0_um: float, wavelength_um: float) -> float:
    w0_m = w0_um * 1e-6
    lambda_m = wavelength_um * 1e-6
    return float(m2 * lambda_m / (np.pi * w0_m) * 1000.0)


def calculate_bpp_um_mrad(w0_um: float, theta_mrad: float) -> float:
    return float(w0_um * theta_mrad)


def calculate_ellipticity(w0_h_um: float, w0_v_um: float) -> float:
    return float(max(w0_h_um, w0_v_um) / min(w0_h_um, w0_v_um))


def calculate_astigmatism_cm(z0_h_cm: float, z0_v_cm: float) -> float:
    return float(abs(z0_h_cm - z0_v_cm))


def calculate_peak_irradiance_per_watt_w_m2(w0_h_um: float, w0_v_um: float) -> float:
    w0_h_m = w0_h_um * 1e-6
    w0_v_m = w0_v_um * 1e-6
    return float(2.0 / (np.pi * w0_h_m * w0_v_m))


def pib_fraction(radius_um: float, w_eff_um: float) -> float:
    return float(1.0 - np.exp(-2.0 * (radius_um / w_eff_um) ** 2))


def gaussian_beam_radius_um(z_cm: np.ndarray, w0_um: float, z0_cm: float, m2: float, wavelength_um: float) -> np.ndarray:
    return np.sqrt(w0_um**2 + ((m2 * wavelength_um * (z_cm - z0_cm) * 1e4) / (np.pi * w0_um)) ** 2)


def plot_beam_profile(
    positions_cm: np.ndarray,
    measured_radii_um: np.ndarray,
    fit_result: M2FitResult,
    wavelength_um: float,
    title: str,
) -> None:
    z_fit_cm = np.linspace(float(np.min(positions_cm)), float(np.max(positions_cm)), 500)
    y_fit_um = gaussian_beam_radius_um(z_fit_cm, fit_result.w0_um, fit_result.z0_cm, fit_result.m2, wavelength_um)

    y_upper_high = y_fit_um * 1.10
    y_lower_high = y_fit_um * 0.90
    y_upper_med = y_fit_um * 1.25
    y_lower_med = y_fit_um * 0.75

    zR_cm = calculate_rayleigh_range_cm(fit_result.m2, fit_result.w0_um, wavelength_um)

    annotation_text = (
        f"$w_0$ = {fit_result.w0_um:.1f} µm\n"
        f"$M^2$ = {fit_result.m2:.2f}\n"
        f"$z_R$ = {zR_cm:.2f} cm\n"
        f"$R^2$ = {fit_result.r_squared:.3f}"
    )

    plt.figure(figsize=(8, 5))
    plt.fill_between(z_fit_cm, y_lower_med, y_upper_med, color="orange", alpha=0.2, label="0.75 ≤ $R^2$ < 0.9")
    plt.fill_between(z_fit_cm, y_lower_high, y_upper_high, color="green", alpha=0.3, label="$R^2$ ≥ 0.9")
    plt.plot(positions_cm, measured_radii_um, "o-", color="black", label="Measured")
    plt.plot(z_fit_cm, y_fit_um, "--", color="red", label="Theoretical Fit")
    plt.title(title)
    plt.xlabel("Position (cm)")
    plt.ylabel("Beam Radius (µm)")
    plt.text(
        0.05,
        0.05,
        annotation_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_summary_table(rows: List[Tuple[str, str, str]]) -> None:
    print("\n" + "=" * 72)
    print("COMPREHENSIVE BEAM PARAMETER SUMMARY")
    print("=" * 72)
    print(f"{'Parameter':<30} {'Value':<25} {'Units':<12}")
    print("-" * 72)
    for param, value, unit in rows:
        print(f"{param:<30} {value:<25} {unit:<12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Beam profiling script converted from BeamChar2 notebook")
    parser.add_argument("folder_path", help="Path to folder containing .hdf5 files")
    parser.add_argument("--base-filename", default="BeamProfile800fixed", help="Base filename prefix")
    parser.add_argument("--file-extension", default=".hdf5", help="File extension")
    parser.add_argument("--wavelength-um", type=float, default=0.8, help="Laser wavelength in µm")
    parser.add_argument("--pixel-pitch-um", type=float, default=2.2, help="Camera pixel pitch in µm/pixel")
    parser.add_argument("--show-profile-fits", action="store_true", help="Show Gaussian fits for each file/profile")
    args = parser.parse_args()

    folder = Path(args.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Invalid folder path: {folder}")

    files = list_sorted_files(folder, args.base_filename, args.file_extension)
    if not files:
        raise FileNotFoundError(
            f"No matching {args.file_extension} files found for base name '{args.base_filename}' in {folder}"
        )

    print(f"Looking for files in: {folder.resolve()}")
    print(f"Matched files: {len(files)}")
    print("Sorted files:", [p.name for _, p in files])

    positions: List[float] = []
    e2_horizontal: List[float] = []
    e2_vertical: List[float] = []
    fwhm_horizontal: List[float] = []
    fwhm_vertical: List[float] = []
    second_moment_h: List[float] = []
    second_moment_v: List[float] = []
    profile_r2_h: List[float] = []
    profile_r2_v: List[float] = []

    for position_cm, file_path in files:
        print(f"\nProcessing: {file_path.name} (z={position_cm:+.1f} cm)")
        with h5py.File(file_path, "r") as f:
            if "image_raw" not in f:
                print("  Skipped (no 'image_raw' dataset)")
                continue
            image = f["image_raw"][()]

        if image.ndim != 2:
            print(f"  Skipped (image_raw is not 2D, shape={image.shape})")
            continue

        sum_cols = image.sum(axis=0)
        sum_rows = image.sum(axis=1)

        fit_h = fit_profile(sum_cols, args.pixel_pitch_um)
        fit_v = fit_profile(sum_rows, args.pixel_pitch_um)

        if args.show_profile_fits:
            for label, fit in (("horizontal", fit_h), ("vertical", fit_v)):
                plt.figure(figsize=(8, 4))
                plt.plot(fit["x"], fit["y"], label="Data")
                plt.plot(fit["x"], fit["y_fit"], "--", label="Fitted Gaussian")
                plt.title(f"{file_path.name}: {label} profile")
                plt.text(
                    0.05,
                    0.95,
                    f"$R^2$ = {fit['r_squared']:.3f}",
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
                plt.legend()
                plt.tight_layout()
                plt.show()

        positions.append(position_cm)
        e2_horizontal.append(fit_h["e2_radius_um"])
        e2_vertical.append(fit_v["e2_radius_um"])
        fwhm_horizontal.append(fit_h["fwhm_um"])
        fwhm_vertical.append(fit_v["fwhm_um"])
        second_moment_h.append(fit_h["second_moment_width_um"])
        second_moment_v.append(fit_v["second_moment_width_um"])
        profile_r2_h.append(fit_h["r_squared"])
        profile_r2_v.append(fit_v["r_squared"])

        print(
            f"  Horizontal: 1/e² radius={fit_h['e2_radius_um']:.2f} µm, "
            f"FWHM={fit_h['fwhm_um']:.2f} µm, D4σ={fit_h['second_moment_width_um']:.2f} µm, R²={fit_h['r_squared']:.4f}"
        )
        print(
            f"  Vertical:   1/e² radius={fit_v['e2_radius_um']:.2f} µm, "
            f"FWHM={fit_v['fwhm_um']:.2f} µm, D4σ={fit_v['second_moment_width_um']:.2f} µm, R²={fit_v['r_squared']:.4f}"
        )

    positions_arr = np.array(positions, dtype=float)
    e2_h_arr = np.array(e2_horizontal, dtype=float)
    e2_v_arr = np.array(e2_vertical, dtype=float)

    m2_h = calculate_m2(positions_arr, e2_h_arr, args.wavelength_um)
    m2_v = calculate_m2(positions_arr, e2_v_arr, args.wavelength_um)

    m2_total = float(np.sqrt(m2_h.m2 * m2_v.m2))
    m2_total_unc = float(
        0.5
        * m2_total
        * np.sqrt((m2_h.m2_uncertainty / m2_h.m2) ** 2 + (m2_v.m2_uncertainty / m2_v.m2) ** 2)
    )

    zR_h_cm = calculate_rayleigh_range_cm(m2_h.m2, m2_h.w0_um, args.wavelength_um)
    zR_v_cm = calculate_rayleigh_range_cm(m2_v.m2, m2_v.w0_um, args.wavelength_um)
    conf_h_cm = 2.0 * zR_h_cm
    conf_v_cm = 2.0 * zR_v_cm

    div_h_mrad = calculate_divergence_half_angle_mrad(m2_h.m2, m2_h.w0_um, args.wavelength_um)
    div_v_mrad = calculate_divergence_half_angle_mrad(m2_v.m2, m2_v.w0_um, args.wavelength_um)

    bpp_h = calculate_bpp_um_mrad(m2_h.w0_um, div_h_mrad)
    bpp_v = calculate_bpp_um_mrad(m2_v.w0_um, div_v_mrad)

    ellipticity = calculate_ellipticity(m2_h.w0_um, m2_v.w0_um)
    astigmatism_cm = calculate_astigmatism_cm(m2_h.z0_cm, m2_v.z0_cm)

    peak_irradiance_per_watt = calculate_peak_irradiance_per_watt_w_m2(m2_h.w0_um, m2_v.w0_um)

    w_eff_um = float(np.sqrt(m2_h.w0_um * m2_v.w0_um))
    r_fwhm_eff_um = float(np.sqrt(np.log(2.0) / 2.0) * w_eff_um)
    pib_fwhm = pib_fraction(r_fwhm_eff_um, w_eff_um)
    pib_1e2 = pib_fraction(w_eff_um, w_eff_um)
    pib_2x = pib_fraction(2.0 * w_eff_um, w_eff_um)

    print_summary_table(
        [
            ("M² Horizontal", f"{m2_h.m2:.3f} ± {m2_h.m2_uncertainty:.3f}", "-"),
            ("M² Vertical", f"{m2_v.m2:.3f} ± {m2_v.m2_uncertainty:.3f}", "-"),
            ("M² Total", f"{m2_total:.3f} ± {m2_total_unc:.3f}", "-"),
            ("Waist Horizontal (w0)", f"{m2_h.w0_um:.2f} ± {m2_h.w0_uncertainty_um:.2f}", "µm"),
            ("Waist Vertical (w0)", f"{m2_v.w0_um:.2f} ± {m2_v.w0_uncertainty_um:.2f}", "µm"),
            ("FWHM Horizontal @ waist", f"{np.sqrt(2*np.log(2))*m2_h.w0_um:.2f}", "µm"),
            ("FWHM Vertical @ waist", f"{np.sqrt(2*np.log(2))*m2_v.w0_um:.2f}", "µm"),
            ("Rayleigh range Horizontal", f"{zR_h_cm:.3f}", "cm"),
            ("Rayleigh range Vertical", f"{zR_v_cm:.3f}", "cm"),
            ("Confocal parameter Horizontal", f"{conf_h_cm:.3f}", "cm"),
            ("Confocal parameter Vertical", f"{conf_v_cm:.3f}", "cm"),
            ("Far-field divergence H", f"{div_h_mrad:.4f}", "mrad"),
            ("Far-field divergence V", f"{div_v_mrad:.4f}", "mrad"),
            ("BPP Horizontal", f"{bpp_h:.4f}", "µm·mrad"),
            ("BPP Vertical", f"{bpp_v:.4f}", "µm·mrad"),
            ("Ellipticity", f"{ellipticity:.4f}", "-"),
            ("Astigmatism", f"{astigmatism_cm:.4f}", "cm"),
            ("Peak irradiance @ waist", f"{peak_irradiance_per_watt:.3e}", "W/m² per W"),
            ("PIB at effective FWHM radius", f"{pib_fwhm*100:.2f}", "%"),
            ("PIB at effective 1/e² radius", f"{pib_1e2*100:.2f}", "%"),
            ("PIB at 2× effective 1/e² radius", f"{pib_2x*100:.2f}", "%"),
            (
                "Second-moment width H (D4σ)",
                f"{np.nanmean(np.array(second_moment_h, dtype=float)):.2f}",
                "µm",
            ),
            (
                "Second-moment width V (D4σ)",
                f"{np.nanmean(np.array(second_moment_v, dtype=float)):.2f}",
                "µm",
            ),
            ("Gaussian fit R² H (avg)", f"{np.nanmean(np.array(profile_r2_h, dtype=float)):.4f}", "-"),
            ("Gaussian fit R² V (avg)", f"{np.nanmean(np.array(profile_r2_v, dtype=float)):.4f}", "-"),
        ]
    )

    plot_beam_profile(positions_arr, e2_h_arr, m2_h, args.wavelength_um, "Horizontal Beam Profile")
    plot_beam_profile(positions_arr, e2_v_arr, m2_v, args.wavelength_um, "Vertical Beam Profile")


if __name__ == "__main__":
    main()
