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

    fig1.suptitle(f"Iter1 white-band and masks: {result.science_path.name}", fontsize=12)
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
    # PAGE 3 — sky spectra + fit + residual
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