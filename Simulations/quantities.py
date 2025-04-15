"""Application of the oxygen diffusion model in Nb.

This module contains functions that make use of the oxygen diffusion model in
Nb to simulate adulteration to its intrinsic properties caused by (spatially
inhomogeneous) interstitial doping.
"""

from typing import Annotated, Sequence
import numpy as np
from scipy import constants


def ell(
    c: float,
    a_0: float = 4.5e-12,
    sigma_0: float = 0.37e-15,
) -> float:
    r"""Nb's electron mean-free-path.

    Calculate the mean-free-path :math:`\ell` of Nb's electrons as a result of
    oxygen doping (e.g., from surface heat-treatments). The calculation makes
    use of two empirical relationships:

    * The linear proportionality between impurity concentration and residual
      resistivity :math:`\rho_{0}`\ .
    * The inverse proportionality between residual resistivity and the
      mean-free-path :math:`\ell`\ .

    Args:
        c: Impurity concentration (at. %).
        a_0: Proportionality coefficient between oxygen concentration and Nb's residual resistivity :math:`\rho_{0}` (\ :math:`\ohm` m ppma\ :sup:`-1`\ ).
        sigma_0: Proportionality coefficient between Nb's residual resistivity :math:`\rho_{0}` its electron mean-free-path (\ :math:`\ohm` m\ :sup:`2`\ ).

    Returns:
        The electron mean-free-path (nm).
    """
    # Define a small epsilon to avoid division by zero
    epsilon = 1e-10

    # Convert the oxygen concentration in at. % to ppma
    stoich_per_at_percent = 1e-2
    ppma_per_stoich = 1e6
    c_ppma = c * stoich_per_at_percent * ppma_per_stoich

    # Clip the denominator to avoid it being too small
    denominator = np.clip(a_0 * c_ppma, epsilon, None)

    ell_m = sigma_0 / denominator  # in meters
    nm_per_m = 1e9
    ell_nm = ell_m * nm_per_m      # convert to nm
    return ell_nm



    # Safeguard against near-zero values in the denominator
    #denominator = np.maximum(a_0 * c_ppma, epsilon)

    # Calculate the mean-free-path (in m)
    # ell_m = sigma_0 / ( a_0 * c_ppma ) 
    # for i, val in enumerate(ell_m):
    #     if np.isfinite(val) == False or val > 1e100:
    #         ell_m[i] = 1e100

    # #print(ell_m)
    # #print(max(ell_m))


    # # Convert the mean-free-path to nm
    # nm_per_m = 1e9
    # return ell_m * nm_per_m



def lambda_eff(
    ell: float,
    lambda_L: float = 27.0,
    xi_0: float = 38.5,
) -> float:
    r"""Effective magnetic penetration depth.

    Effective magnetic penetration depth :math:`\lambda_{\mathrm{eff.}` for an
    impure superconductor (at 0 K).

    Args:
        ell: Electron mean-free-path :math:`\ell` (nm).
        lambda_L: London penetration depth :math:`\lambda_{L}` (nm).
        xi_0: Pippard/Bardeen-Cooper-Schrieffer coherence length (nm).

    Returns:
        The effective magnetic penetration depth (nm).
    """

    # Note: the factor pi/2 is necessary to be in accord with BCS theory
    # see, e.g.:
    #
    # P. B. Miller, "Penetration Depth in Impure Superconductors",
    # Phys. Rev. 113, 1209 (1959).
    # https://doi.org/10.1103/PhysRev.113.1209
    #
    # J. Halbritter, "On the penetration of the magnetic field into a
    # superconductor", Z. Phys. 243, 201–219 (1971).
    # https://doi.org/10.1007/BF01394851
    return lambda_L * np.sqrt(1.0 + (np.pi / 2.0) * (xi_0 / ell))


def B(
    z_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    penetration_depth_nm: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None] = 0.0,
    demagnetization_factor: Annotated[float, 0:1] = 0.0,
) -> Sequence[float]:
    """Meissner screening profile for the simple London model.
    
    Args:
        z_nm: Depth below the surface (nm).
        penetration_depth_nm: Magnetic penetration depth (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Non-superconducting dead layer thickness (nm).
        demagnetization_factor: Effective demagnetization factor.
    
    Returns:
        The magnetic field as a function of depth (G).
    """

    effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

    return effective_field_G * np.exp(-z_nm / penetration_depth_nm)


def J(
    z_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    penetration_depth_nm: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None] = 0.0,
    demagnetization_factor: Annotated[float, 0:1] = 0.0,
) -> Sequence[float]:
    """Meissner current density for the simple London model.
    
    Args:
        z_nm: Depth below the surface (nm).
        penetration_depth_nm: Magnetic penetration depth (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Non-superconducting dead layer thickness (nm).
        demagnetization_factor: Effective demagnetization factor.
    
    Returns:
        The current density as a function of the depth (A m^-2).
    """

    # calculate the prefactor for the conversion
    G_per_T = 1e4
    # nm_per_m = 1e9
    m_per_nm = 1e-9
    mu_0 = constants.value("vacuum mag. permeability") * G_per_T

    j_0 = -1.0 / mu_0

    # correct the depth for the dead layer
    z_corr_nm = z_nm - dead_layer_nm

    return (
        j_0
        * (-1.0 / penetration_depth_nm / m_per_nm)
        * B(
            z_nm,
            applied_field_G,
            penetration_depth_nm,
            dead_layer_nm,
            demagnetization_factor,
        )
    )
