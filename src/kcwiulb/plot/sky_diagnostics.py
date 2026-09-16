from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def format_fit_params(params, pcov=None) -> str:
    """
    Format fit parameters for display in the diagnostic figure.

    For blue iteration 1, the fitted model is:
        (a0 + a1 * wl) * sky1 + (b0 + b1 * wl) * sky2 + c0 + c1 * wl
    """
    names = ["a0", "a1", "b0", "b1", "c0", "c1"]
    lines = []

    for i, p in enumerate(params):
        if pcov is not None and np.ndim(pcov) == 2 and i < pcov.shape[0]:
            err = np.sqrt(np.abs(pcov[i, i]))
            lines.append(f"{names[i]} = {p:.3e} ± {err:.1e}")
        else:
            lines.append(f"{names[i]} = {p:.3e}")

    return "\n".join(lines)


def format_fit_params_four(params, pcov=None, label: str = "params") -> str:
    """
    Format 4-parameter fit results for iter2.
    """
    names = ["a", "b", "c", "d"]
    lines = [label]

    for i, p in enumerate(params):
        if pcov is not None and np.ndim(pcov) == 2 and i < pcov.shape[0]:
            err = np.sqrt(np.abs(pcov[i, i]))
            lines.append(f"{names[i]} = {p:.3e} ± {err:.1e}")
        else:
            lines.append(f"{names[i]} = {p:.3e}")

    return "\n".join(lines)


def plot_blue_iter1_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    residual_zoom_ylim: tuple[float, float] = (-0.002, 0.002),
) -> None:
    """
    Save blue iteration 1 diagnostics as a multi-page PDF.

    Pages:
    1. White-band images and masks
    2. Median spectra
    3. Sky spectra + fit + residual + fitted parameters
    4. Zoomed residual
    """
    pdf = None
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    fig1 = plt.figure(figsize=(9, 5))

    titles = ["Science", "Sky 1", "Sky 2"]
    images = [
        result.science_whiteband,
        result.sky1_whiteband,
        result.sky2_whiteband,
    ]
    masks = [
        result.science_mask,
        result.sky1_mask,
        result.sky2_mask,
    ]

    for i in range(3):
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(
            images[i],
            origin="lower",
            cmap="RdBu_r",
            aspect=0.2,
            vmin=whiteband_vmin,
            vmax=whiteband_vmax,
        )
        ax.set_title(f"{titles[i]} white-band")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 3, i + 4)
        ax.imshow(
            masks[i],
            origin="lower",
            cmap="gray",
            aspect=0.2,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{titles[i]} mask")
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.suptitle(f"Iter1 white-band and masks: {result.science_path.name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.science_spec, label="science")
    ax.plot(result.wavelength, result.sky1_spec, label="sky1")
    ax.plot(result.wavelength, result.sky2_spec, label="sky2")

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend()
    ax.set_title("Median sky spectra")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    fig3 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.science_spec, label="science", lw=1.2)
    ax.plot(result.wavelength, result.sky1_spec, label="sky1", lw=1.0, alpha=0.8)
    ax.plot(result.wavelength, result.sky2_spec, label="sky2", lw=1.0, alpha=0.8)
    ax.plot(result.wavelength, result.model_spec, "--", label="model", lw=1.2)
    ax.plot(result.wavelength, result.residual_spec, label="residual", lw=1.0)

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", fontsize=9)

    model_text = (
        "Model:\n"
        "(a0 + a1·wl)·sky1 + (b0 + b1·wl)·sky2 + c0 + c1·wl"
    )

    param_text = model_text + "\n\n" + format_fit_params(
        result.params, getattr(result, "pcov", None)
    )

    if hasattr(result, "chi2") and result.chi2 is not None:
        param_text += f"\n\nchi2 = {result.chi2:.3e}"

    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.85,
        ),
    )

    ax.set_title("Sky spectra, fit, and residual")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    fig4 = plt.figure(figsize=(9, 2))
    ax = plt.gca()

    ax.plot(result.wavelength, result.residual_spec, lw=1)
    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.set_ylim(residual_zoom_ylim)
    ax.set_title("Residual spectrum (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)


def plot_blue_selfsky_iter1_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    residual_zoom_ylim: tuple[float, float] = (-0.002, 0.002),
) -> None:
    """
    Save blue self-sky diagnostics as a multi-page PDF.

    Pages:
    1. White-band image, source mask, and final sky mask
    2. Extracted self-sky spectrum
    3. Sky spectrum and post-subtraction residual
    4. Zoomed residual spectrum

    Mask convention:
        source_mask = True for pixels excluded by the QSO/source radius
        sky_mask    = True for all pixels excluded from the sky estimate

    Therefore ~sky_mask gives the pixels actually used to construct
    the self-sky spectrum.
    """
    pdf = None

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — spatial sky selection
    # ========================================================
    fig1 = plt.figure(figsize=(11, 5))

    # --------------------------------------------------------
    # White-band image
    # --------------------------------------------------------
    ax = plt.subplot(1, 3, 1)

    ax.imshow(
        result.whiteband,
        origin="lower",
        cmap="RdBu_r",
        aspect=0.2,
        vmin=whiteband_vmin,
        vmax=whiteband_vmax,
    )

    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="k",
    )

    ax.set_title("Science white-band")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --------------------------------------------------------
    # QSO/source exclusion mask
    #
    # True = excluded because it lies within the requested
    # source-protection radius.
    # --------------------------------------------------------
    ax = plt.subplot(1, 3, 2)

    ax.imshow(
        result.source_mask,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )

    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )

    ax.set_title(
        f"Source mask\n"
        f"R = {result.source_mask_radius_arcsec:.1f}\""
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --------------------------------------------------------
    # Final pixels used for self-sky
    #
    # sky_mask:
    #   True  = excluded
    #   False = used for sky
    #
    # Display ~sky_mask so white pixels are the actual pixels
    # contributing to the sky spectrum.
    # --------------------------------------------------------
    ax = plt.subplot(1, 3, 3)

    sky_used = ~result.sky_mask

    ax.imshow(
        sky_used,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )

    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )

    n_total = result.sky_mask.size
    n_sky = np.count_nonzero(sky_used)
    sky_fraction = n_sky / n_total

    ax.set_title(
        f"Pixels used for sky\n"
        f"N = {n_sky}/{n_total} ({sky_fraction:.1%})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig1.suptitle(
        f"Blue self-sky spatial selection: {result.science_path.name}",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — extracted self-sky spectrum
    # ========================================================
    fig2 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.sky_spec,
        label="self-sky",
        lw=1.0,
    )

    finite_uncert = np.isfinite(result.sky_uncert)

    if np.any(finite_uncert):
        lower = result.sky_spec - result.sky_uncert
        upper = result.sky_spec + result.sky_uncert

        ax.fill_between(
            result.wavelength,
            lower,
            upper,
            where=finite_uncert,
            alpha=0.2,
            label="±1σ",
        )

    ax.axvline(
        x=result.wavgood0,
        c="k",
        linestyle="--",
        alpha=0.7,
        label="WAVGOOD0",
    )

    ax.axvline(
        x=result.wavgood1,
        c="k",
        linestyle="--",
        alpha=0.7,
        label="WAVGOOD1",
    )

    ax.legend(loc="upper right", fontsize=9)

    ax.set_title("Extracted self-sky spectrum")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — sky spectrum + residual
    # ========================================================
    fig3 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.sky_spec,
        label="self-sky",
        lw=1.0,
    )

    ax.plot(
        result.wavelength,
        result.residual_spec,
        label="post-subtraction sky residual",
        lw=1.0,
    )

    ax.axhline(
        0,
        c="k",
        linestyle=":",
        alpha=0.7,
    )

    ax.axvline(
        x=result.wavgood0,
        c="k",
        linestyle="--",
        alpha=0.7,
    )

    ax.axvline(
        x=result.wavgood1,
        c="k",
        linestyle="--",
        alpha=0.7,
    )

    ax.legend(loc="upper right", fontsize=9)

    ax.set_title("Self-sky subtraction")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — zoomed residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 2.5))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.residual_spec,
        lw=1.0,
    )

    ax.axhline(
        0,
        c="k",
        linestyle=":",
        alpha=0.7,
    )

    ax.axvline(
        x=result.wavgood0,
        c="k",
        linestyle="--",
        alpha=0.7,
    )

    ax.axvline(
        x=result.wavgood1,
        c="k",
        linestyle="--",
        alpha=0.7,
    )

    ax.set_ylim(residual_zoom_ylim)

    ax.set_title("Post-subtraction sky residual (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)


def plot_blue_iter2_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    continuum_ylim: tuple[float, float] = (-0.01, 0.06),
    residual_zoom_ylim: tuple[float, float] = (-0.005, 0.005),
) -> None:
    """
    Save blue iteration 2 diagnostics as a multi-page PDF.

    Pages:
    1. White-band images and masks for science + 4 skies
    2. Region-1 median sky spectra with continuum filtering
    3. Residual model and fitted parameters for all wavelength regions
    4. Zoomed residual with all region boundaries
    """
    pdf = None
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — whiteband images + masks
    # ========================================================
    fig1 = plt.figure(figsize=(12, 5))

    titles = ["Science", "Sky 1", "Sky 2", "Sky 3", "Sky 4"]
    images = [
        result.science_whiteband,
        result.sky1_whiteband,
        result.sky2_whiteband,
        result.sky3_whiteband,
        result.sky4_whiteband,
    ]
    masks = [
        result.science_mask,
        result.sky1_mask,
        result.sky2_mask,
        result.sky3_mask,
        result.sky4_mask,
    ]

    for i in range(5):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(
            images[i],
            origin="lower",
            cmap="RdBu_r",
            aspect=0.2,
            vmin=whiteband_vmin,
            vmax=whiteband_vmax,
        )
        ax.set_title(f"{titles[i]} white-band")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 5, i + 6)
        ax.imshow(
            masks[i],
            origin="lower",
            cmap="gray",
            aspect=0.2,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{titles[i]} mask")
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.suptitle(f"Iter2 white-band and masks: {result.science_path.name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — median sky with continuum filtering
    # ========================================================
    fig2 = plt.figure(figsize=(9, 8))

    region_specs = [
        (result.science_spec, result.science_spec_cont, "science"),
        (result.sky1_spec, result.sky1_spec_cont, "sky1"),
        (result.sky2_spec, result.sky2_spec_cont, "sky2"),
        (result.sky3_spec, result.sky3_spec_cont, "sky3"),
        (result.sky4_spec, result.sky4_spec_cont, "sky4"),
    ]

    for i, (spec, cont, label) in enumerate(region_specs, start=1):
        ax = plt.subplot(5, 1, i)
        ax.plot(result.wavelength, spec, label=label)
        ax.plot(result.wavelength, cont, label=f"{label} continuum")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(continuum_ylim)
        if i < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")

    fig2.suptitle("Region-1 median sky spectra and continuum filtering", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — residual model + fitted results
    # ========================================================
    fig3 = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.sky1_spec, label="science sky", lw=1.1)
    ax.plot(result.wavelength, result.fit_residual, label="fit_residual", lw=1.2)

    for idx, region_res in enumerate(result.fit_residual_regions, start=1):
        ax.plot(
            result.wavelength,
            region_res,
            lw=0.9,
            alpha=0.8,
            label=f"region {idx} residual",
        )

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7, label="WAVGOOD0")
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7, label="WAVGOOD1")

    for i, (_, (w0, w1)) in enumerate(zip(result.region_bounds, result.region_wavelength_bounds), start=1):
        if i < len(result.region_wavelength_bounds):
            ax.axvline(x=w1, c="gray", linestyle=":", alpha=0.8)

    ax.legend(loc="upper right", fontsize=8)

    model_text = "Model:\na·sky2 + b·sky3 + c·sky4 + d·sky5"

    region_text_blocks = []
    for i, (params, pcov, (w0, w1)) in enumerate(
        zip(result.params_list, result.pcov_list, result.region_wavelength_bounds),
        start=1,
    ):
        label = f"region {i}: {w0:.0f}-{w1:.0f}"
        region_text_blocks.append(format_fit_params_four(params, pcov, label=label))

    param_text = model_text + "\n\n" + "\n\n".join(region_text_blocks)

    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.85,
        ),
    )

    ax.set_title("Residual model and fitted parameters")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — zoomed residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 2.5))
    ax = plt.gca()

    ax.plot(result.wavelength, result.fit_residual, lw=1.2, label="fit_residual")

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    for i, (_, (w0, w1)) in enumerate(zip(result.region_bounds, result.region_wavelength_bounds), start=1):
        ax.axvline(x=w0, c="gray", linestyle=":", alpha=0.7)
        if i == len(result.region_wavelength_bounds):
            ax.axvline(x=w1, c="gray", linestyle=":", alpha=0.7)

    ax.set_ylim(residual_zoom_ylim)
    ax.set_title("Residual spectrum (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)

def plot_red_iter1_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    residual_zoom_ylim: tuple[float, float] = (-0.002, 0.002),
) -> None:
    """
    Save red iteration 1 diagnostics as a multi-page PDF.

    Pages:
    1. White-band images and masks
    2. Median spectra
    3. Sky spectra + fit + residual + fitted parameters
    4. Zoomed residual
    """
    pdf = None
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — whiteband images + masks
    # ========================================================
    fig1 = plt.figure(figsize=(9, 5))

    titles = ["Science", "Sky 1", "Sky 2"]
    images = [
        result.science_whiteband,
        result.sky1_whiteband,
        result.sky2_whiteband,
    ]
    masks = [
        result.science_mask,
        result.sky1_mask,
        result.sky2_mask,
    ]

    for i in range(3):
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(
            images[i],
            origin="lower",
            cmap="RdBu_r",
            aspect=0.2,
            vmin=whiteband_vmin,
            vmax=whiteband_vmax,
        )
        ax.set_title(f"{titles[i]} white-band")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(2, 3, i + 4)
        ax.imshow(
            masks[i],
            origin="lower",
            cmap="gray",
            aspect=0.2,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{titles[i]} mask")
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.suptitle(f"Red iter1 white-band and masks: {result.science_path.name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — median spectra
    # ========================================================
    fig2 = plt.figure(figsize=(8, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.science_spec, label="science")
    ax.plot(result.wavelength, result.sky1_spec, label="sky1")
    ax.plot(result.wavelength, result.sky2_spec, label="sky2")

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend()
    ax.set_title("Median sky spectra")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — fit and residual
    # ========================================================
    fig3 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.science_spec, label="science", lw=1.2)
    ax.plot(result.wavelength, result.sky1_spec, label="sky1", lw=1.0, alpha=0.8)
    ax.plot(result.wavelength, result.sky2_spec, label="sky2", lw=1.0, alpha=0.8)
    ax.plot(result.wavelength, result.model_spec, "--", label="model", lw=1.2)
    ax.plot(result.wavelength, result.residual_spec, label="residual", lw=1.0)

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", fontsize=9)

    model_text = (
        "Model:\n"
        "(a0 + a1·wl)·sky1 + (b0 + b1·wl)·sky2 + c0 + c1·wl"
    )

    pcov = getattr(result, "covariance", None)
    param_text = model_text + "\n\n" + format_fit_params(result.params, pcov)

    if hasattr(result, "chi2") and result.chi2 is not None:
        param_text += f"\n\nchi2 = {result.chi2:.3e}"

    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.85,
        ),
    )

    ax.set_title("Sky spectra, fit, and residual")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — zoomed residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 2))
    ax = plt.gca()

    ax.plot(result.wavelength, result.residual_spec, lw=1)
    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.set_ylim(residual_zoom_ylim)
    ax.set_title("Residual spectrum (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)


def plot_red_iter2_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    residual_zoom_ylim: tuple[float, float] = (-0.002, 0.002),
) -> None:
    """
    Save red iteration 2 diagnostics as a multi-page PDF.

    This plotting function assumes the subtraction code already computed
    notebook-faithful white-band images and masks.

    Pages:
    1. White-band images and masks
    2. Median sky spectra
    3. Sky spectra, fit, and residual
    4. Residual spectrum (zoomed)
    """
    pdf = None
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — white-band images + masks
    # matches notebook layout exactly
    # ========================================================
    fig1 = plt.figure(figsize=(9, 5))

    ax = plt.subplot(2, 3, 1)
    ax.imshow(
        result.science_whiteband,
        origin="lower",
        cmap="RdBu_r",
        aspect=0.2,
        vmin=whiteband_vmin,
        vmax=whiteband_vmax,
    )
    ax.set_title("Science white-band")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 3, 4)
    ax.imshow(
        result.science_mask_2d,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Science mask")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 3, 2)
    ax.imshow(
        result.sky1_whiteband,
        origin="lower",
        cmap="RdBu_r",
        aspect=0.2,
        vmin=whiteband_vmin,
        vmax=whiteband_vmax,
    )
    ax.set_title("Sky 1 white-band")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 3, 5)
    ax.imshow(
        result.sky1_mask_2d,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Sky 1 mask")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 3, 3)
    ax.imshow(
        result.sky2_whiteband,
        origin="lower",
        cmap="RdBu_r",
        aspect=0.2,
        vmin=whiteband_vmin,
        vmax=whiteband_vmax,
    )
    ax.set_title("Sky 2 white-band")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(2, 3, 6)
    ax.imshow(
        result.sky2_mask_2d,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Sky 2 mask")
    ax.set_xticks([])
    ax.set_yticks([])

    fig1.suptitle(f"Red iter2 white-band and masks: {result.science_path.name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — median spectra
    # ========================================================
    fig2 = plt.figure(figsize=(8, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.science_spec, label="science")
    ax.plot(result.wavelength, result.sky1_spec, label="sky1")
    ax.plot(result.wavelength, result.sky2_spec, label="sky2")

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend()
    ax.set_title("Median sky spectra")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — fit and residual
    # ========================================================
    fig3 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(result.wavelength, result.science_spec, label="science", lw=1.2)
    ax.plot(result.wavelength, result.sky1_spec, label="sky1", lw=1.0, alpha=0.8)
    ax.plot(result.wavelength, result.sky2_spec, label="sky2", lw=1.0, alpha=0.8)
    ax.plot(result.wavelength, result.model_spec, "--", label="model", lw=1.2)
    ax.plot(result.wavelength, result.residual_spec, label="residual", lw=1.0)

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", fontsize=9)

    model_text = (
        "Model:\n"
        "(a0 + a1·wl)·sky1 + (b0 + b1·wl)·sky2 + c0 + c1·wl"
    )

    pcov = getattr(result, "covariance", None)
    param_text = model_text + "\n\n" + format_fit_params(result.params, pcov)

    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.85,
        ),
    )

    ax.set_title("Sky spectra, fit, and residual")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — zoomed residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 2))
    ax = plt.gca()

    ax.plot(result.wavelength, result.residual_spec, lw=1)
    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.set_ylim(residual_zoom_ylim)
    ax.set_title("Residual spectrum (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)


def plot_red_iter3_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    continuum_ylim: tuple[float, float] = (0.0, 0.6),
    figure4_ylim: tuple[float, float] = (-0.004, 0.004),
) -> None:
    """
    Save red iteration 3 diagnostics as a multi-page PDF.

    Pages:
    1. White-band images and masks for science + 4 skies
    2. Median spectra with continuum filtering
    3. Residual model and fitted parameters for all wavelength regions
    4. Zoomed residual with wavelength-region boundaries
    """
    pdf = None
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — white-band images + masks
    # ========================================================
    fig1 = plt.figure(figsize=(9, 5))

    titles = ["Science", "Sky 1", "Sky 2", "Sky 3", "Sky 4"]
    images = [
        result.science_whiteband,
        result.sky1_whiteband,
        result.sky2_whiteband,
        result.sky3_whiteband,
        result.sky4_whiteband,
    ]
    masks = [
        result.science_mask,
        result.sky1_mask,
        result.sky2_mask,
        result.sky3_mask,
        result.sky4_mask,
    ]

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            images[i],
            origin="lower",
            cmap="RdBu_r",
            aspect=0.2,
            vmin=whiteband_vmin,
            vmax=whiteband_vmax,
        )
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 5, i + 6)
        plt.imshow(
            masks[i],
            origin="lower",
            cmap="gray",
            aspect=0.2,
            vmin=0,
            vmax=1,
        )
        plt.xticks([])
        plt.yticks([])

    fig1.suptitle(f"Red iter3 white-band and masks: {result.science_path.name}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — median spectra + continuum filtering
    # ========================================================
    fig2 = plt.figure(figsize=(9, 8))

    specs = [
        (result.science_spec, result.science_spec_cont, "c1"),
        (result.sky1_spec, result.sky1_spec_cont, "c2"),
        (result.sky2_spec, result.sky2_spec_cont, "c3"),
        (result.sky3_spec, result.sky3_spec_cont, "c4"),
        (result.sky4_spec, result.sky4_spec_cont, "c5"),
    ]

    for i, (spec, cont, label) in enumerate(specs, start=1):
        ax = plt.subplot(5, 1, i)
        ax.plot(result.wavelength, spec, label=label)
        ax.plot(result.wavelength, cont, label=f"{label} continuum")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(continuum_ylim)
        ax.set_ylabel("Flux")

        if i < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Wavelength")

    fig2.suptitle("Median sky spectra and continuum filtering", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — residual model + fitted parameters
    # similar to blue iter2
    # ========================================================
    fig3 = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    model_residual = np.zeros_like(result.wavelength, dtype=float)

    for (i0, i1), params in zip(result.region_bounds, result.params_list):
        model_residual[i0:i1] = (
            params[0] * result.sky1_spec_res[i0:i1]
            + params[1] * result.sky2_spec_res[i0:i1]
            + params[2] * result.sky3_spec_res[i0:i1]
            + params[3] * result.sky4_spec_res[i0:i1]
        )

    ax.plot(result.wavelength, result.science_spec_res, label="science residual", lw=1.1)
    ax.plot(result.wavelength, model_residual, "--", label="model residual", lw=1.2)
    ax.plot(result.wavelength, result.fit_residual, label="fit_residual", lw=1.2)

    for idx, region_res in enumerate(result.fit_residual_regions, start=1):
        ax.plot(
            result.wavelength,
            region_res,
            lw=0.9,
            alpha=0.7,
            label=f"region {idx} residual",
        )

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7, label="WAVGOOD0")
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7, label="WAVGOOD1")

    for i, (_, (w0, w1)) in enumerate(
        zip(result.region_bounds, result.region_wavelength_bounds), start=1
    ):
        if i < len(result.region_wavelength_bounds):
            ax.axvline(x=w1, c="gray", linestyle=":", alpha=0.8)

    ax.legend(loc="upper right", fontsize=8)

    model_text = "Model:\na·sky1 + b·sky2 + c·sky3 + d·sky4"

    region_text_blocks = []
    for i, (params, pcov, (w0, w1)) in enumerate(
        zip(result.params_list, result.pcov_list, result.region_wavelength_bounds),
        start=1,
    ):
        label = f"region {i}: {w0:.0f}-{w1:.0f}"
        region_text_blocks.append(format_fit_params_four(params, pcov, label=label))

    param_text = model_text + "\n\n" + "\n\n".join(region_text_blocks)

    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.85,
        ),
    )

    ax.set_title("Sky spectra, fit, and residual")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — zoomed residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 2.5))
    ax = plt.gca()

    ax.plot(result.wavelength, result.fit_residual, lw=1.2, label="fit_residual")

    ax.axvline(x=result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(x=result.wavgood1, c="k", linestyle="--", alpha=0.7)

    for i, (_, (w0, w1)) in enumerate(
        zip(result.region_bounds, result.region_wavelength_bounds), start=1
    ):
        ax.axvline(x=w0, c="gray", linestyle=":", alpha=0.7)
        if i == len(result.region_wavelength_bounds):
            ax.axvline(x=w1, c="gray", linestyle=":", alpha=0.7)

    ax.set_ylim(figure4_ylim)
    ax.set_title("Residual spectrum (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)


def plot_red_selfsky_iter1_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    residual_zoom_ylim: tuple[float, float] = (-0.002, 0.002),
) -> None:
    """
    Save red self-sky iter1 diagnostics as a multi-page PDF.

    Pages:
    1. White-band image, QSO/source mask, and pixels used for sky
    2. Extracted self-sky spectrum
    3. Self-sky spectrum and post-subtraction residual
    4. Zoomed residual spectrum

    Mask convention:
        source_mask = True inside the QSO/source exclusion region
        bright_mask = True for additional sigma-clipped sources
        sky_mask    = source_mask | bright_mask

    Therefore ~sky_mask gives the spatial pixels used to construct
    the iter1 self-sky spectrum.
    """
    pdf = None

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — spatial self-sky selection
    # ========================================================
    fig1 = plt.figure(figsize=(11, 5))

    ax = plt.subplot(1, 3, 1)
    ax.imshow(
        result.whiteband,
        origin="lower",
        cmap="RdBu_r",
        aspect=0.2,
        vmin=whiteband_vmin,
        vmax=whiteband_vmax,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="k",
    )
    ax.set_title("Science white-band")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(
        result.source_mask,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )
    ax.set_title(
        f"Source mask\n"
        f"R = {result.source_mask_radius_arcsec:.1f}\""
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = plt.subplot(1, 3, 3)

    sky_used = ~result.sky_mask
    n_total = result.sky_mask.size
    n_sky = np.count_nonzero(sky_used)

    ax.imshow(
        sky_used,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )
    ax.set_title(
        f"Pixels used for sky\n"
        f"N = {n_sky}/{n_total} ({n_sky / n_total:.1%})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig1.suptitle(
        f"Red self-sky iter1 spatial selection: {result.science_path.name}",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — extracted self-sky spectrum
    # ========================================================
    fig2 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.sky_spec,
        label="self-sky",
        lw=1.0,
    )

    finite_uncert = np.isfinite(result.sky_uncert)

    if np.any(finite_uncert):
        lower = result.sky_spec - result.sky_uncert
        upper = result.sky_spec + result.sky_uncert

        ax.fill_between(
            result.wavelength,
            lower,
            upper,
            where=finite_uncert,
            alpha=0.2,
            label="±1σ",
        )

    ax.axvline(
        result.wavgood0,
        c="k",
        linestyle="--",
        alpha=0.7,
        label="WAVGOOD0",
    )
    ax.axvline(
        result.wavgood1,
        c="k",
        linestyle="--",
        alpha=0.7,
        label="WAVGOOD1",
    )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Red iter1 extracted self-sky spectrum")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — sky + residual
    # ========================================================
    fig3 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.sky_spec,
        label="self-sky",
        lw=1.0,
    )
    ax.plot(
        result.wavelength,
        result.residual_spec,
        label="post-subtraction sky residual",
        lw=1.0,
    )

    ax.axhline(0, c="k", linestyle=":", alpha=0.7)
    ax.axvline(result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Red self-sky iter1 subtraction")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — zoomed residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 2.5))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.residual_spec,
        lw=1.0,
    )

    ax.axhline(0, c="k", linestyle=":", alpha=0.7)
    ax.axvline(result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.set_ylim(residual_zoom_ylim)
    ax.set_title("Red self-sky iter1 residual (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig4)


def plot_red_selfsky_iter2_diagnostics(
    result,
    savepath: str | Path | None = None,
    show: bool = False,
    whiteband_vmin: float = -1,
    whiteband_vmax: float = 80,
    residual_zoom_ylim: tuple[float, float] = (-0.002, 0.002),
) -> None:
    """
    Save red self-sky iter2 diagnostics as a multi-page PDF.

    Iter2 goes back to the original unsubtracted cube and rebuilds the
    self-sky spectrum while excluding CR voxels identified after iter1.

    Pages:
    1. White-band image, spatial source mask, and spatial sky pixels
    2. CR-mask diagnostics
    3. Improved iter2 self-sky spectrum
    4. Self-sky spectrum and post-subtraction residual
    5. Zoomed residual spectrum
    """
    pdf = None

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(savepath.with_suffix(".pdf"))

    # ========================================================
    # PAGE 1 — spatial self-sky selection
    # ========================================================
    fig1 = plt.figure(figsize=(11, 5))

    ax = plt.subplot(1, 3, 1)
    ax.imshow(
        result.whiteband,
        origin="lower",
        cmap="RdBu_r",
        aspect=0.2,
        vmin=whiteband_vmin,
        vmax=whiteband_vmax,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="k",
    )
    ax.set_title("Iter1-subtracted white-band")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(
        result.source_mask,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )
    ax.set_title(
        f"Source mask\n"
        f"R = {result.source_mask_radius_arcsec:.1f}\""
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = plt.subplot(1, 3, 3)

    sky_used = ~result.sky_mask_2d
    n_total = result.sky_mask_2d.size
    n_sky = np.count_nonzero(sky_used)

    ax.imshow(
        sky_used,
        origin="lower",
        cmap="gray",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )
    ax.set_title(
        f"Spatial pixels available for sky\n"
        f"N = {n_sky}/{n_total} ({n_sky / n_total:.1%})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig1.suptitle(
        f"Red self-sky iter2 spatial selection: {result.science_path.name}",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if pdf is not None:
        pdf.savefig(fig1)
        plt.close(fig1)
    elif show:
        plt.show()
    else:
        plt.close(fig1)

    # ========================================================
    # PAGE 2 — CR mask diagnostics
    #
    # CR masking is 3D, so collapse along wavelength:
    #
    #   left: number of CR-masked wavelengths per spaxel
    #   right: fraction of wavelengths retained in final
    #          self-sky estimate
    # ========================================================
    fig2 = plt.figure(figsize=(10, 5))

    cr_count = np.sum(
        result.cr_mask_3d,
        axis=0,
    )

    ax = plt.subplot(1, 2, 1)
    im = ax.imshow(
        cr_count,
        origin="lower",
        aspect=0.2,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )
    ax.set_title(
        f"CR-masked wavelengths per spaxel\n"
        f"Total CR voxels = {int(result.cr_mask_3d.sum())}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="N masked wavelengths")

    valid_fraction = np.mean(
        ~result.master_mask_3d,
        axis=0,
    )

    ax = plt.subplot(1, 2, 2)
    im = ax.imshow(
        valid_fraction,
        origin="lower",
        aspect=0.2,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        result.qso_x,
        result.qso_y,
        marker="+",
        s=100,
        linewidths=1.5,
        c="r",
    )
    ax.set_title("Fraction of wavelengths usable for self-sky")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Valid fraction")

    fig2.suptitle(
        "Red self-sky iter2 CR masking",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if pdf is not None:
        pdf.savefig(fig2)
        plt.close(fig2)
    elif show:
        plt.show()
    else:
        plt.close(fig2)

    # ========================================================
    # PAGE 3 — improved self-sky spectrum
    # ========================================================
    fig3 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.sky_spec,
        label="iter2 self-sky",
        lw=1.0,
    )

    finite_uncert = np.isfinite(result.sky_uncert)

    if np.any(finite_uncert):
        lower = result.sky_spec - result.sky_uncert
        upper = result.sky_spec + result.sky_uncert

        ax.fill_between(
            result.wavelength,
            lower,
            upper,
            where=finite_uncert,
            alpha=0.2,
            label="±1σ",
        )

    ax.axvline(
        result.wavgood0,
        c="k",
        linestyle="--",
        alpha=0.7,
        label="WAVGOOD0",
    )
    ax.axvline(
        result.wavgood1,
        c="k",
        linestyle="--",
        alpha=0.7,
        label="WAVGOOD1",
    )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Red iter2 improved self-sky spectrum")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig3)
        plt.close(fig3)
    elif show:
        plt.show()
    else:
        plt.close(fig3)

    # ========================================================
    # PAGE 4 — sky + residual
    # ========================================================
    fig4 = plt.figure(figsize=(9, 4))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.sky_spec,
        label="iter2 self-sky",
        lw=1.0,
    )
    ax.plot(
        result.wavelength,
        result.residual_spec,
        label="post-subtraction sky residual",
        lw=1.0,
    )

    ax.axhline(0, c="k", linestyle=":", alpha=0.7)
    ax.axvline(result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Red self-sky iter2 subtraction")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig4)
        plt.close(fig4)
    elif show:
        plt.show()
    else:
        plt.close(fig4)

    # ========================================================
    # PAGE 5 — zoomed residual
    # ========================================================
    fig5 = plt.figure(figsize=(9, 2.5))
    ax = plt.gca()

    ax.plot(
        result.wavelength,
        result.residual_spec,
        lw=1.0,
    )

    ax.axhline(0, c="k", linestyle=":", alpha=0.7)
    ax.axvline(result.wavgood0, c="k", linestyle="--", alpha=0.7)
    ax.axvline(result.wavgood1, c="k", linestyle="--", alpha=0.7)

    ax.set_ylim(residual_zoom_ylim)
    ax.set_title("Red self-sky iter2 residual (zoomed)")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig5)
        plt.close(fig5)
        pdf.close()
    elif show:
        plt.show()
    else:
        plt.close(fig5)