#!/usr/bin/env python3
"""
Deuteranopia simulation of VHP4Safety palette colors.

Methodology:
- Vienot, Brettel & Mollon 1999 deuteranopia simulation matrix
- sRGB linearization (IEC 61966-2-1 standard)
- CIEDE2000 delta E (Sharma, Wu, Dalal 2005)
- Confusability threshold: delta E < 10

Output: Markdown-formatted tables for COLORBLIND-FINDINGS.md
"""

import numpy as np
from itertools import combinations

# ============================================================
# VHP4Safety palette colors (from plots/shared.py BRAND_COLORS)
# ============================================================

PALETTE_COLORS = {
    'primary': '#29235C',
    'magenta': '#E6007E',
    'blue': '#307BBF',
    'light_blue': '#009FE3',
    'orange': '#EB5B25',
    'sky_blue': '#93D5F6',
    'deep_magenta': '#9A1C57',
    'teal': '#45A6B2',
    'warm_pink': '#B81178',
    'dark_teal': '#005A6C',
    'violet': '#64358C',
}

OECD_STATUS_COLORS = {
    'EAGMST Under Review': '#307BBF',
    'Under Development': '#009FE3',
    'TFHA/WNT Endorsed': '#29235C',
    'WNT Endorsed': '#E6007E',
    'Approved': '#EB5B25',
    'No Status': '#999999',
    'EAGMST Under Development': '#45A6B2',
    'Not OECD': '#93D5F6',
}

TYPE_COLORS = {
    'Essential': '#29235C',
    'Metadata': '#E6007E',
    'Content': '#EB5B25',
    'Context': '#93D5F6',
    'Assessment': '#307BBF',
    'Structure': '#45A6B2',
}

DELTA_E_THRESHOLD = 10

# ============================================================
# sRGB linearization
# ============================================================

def srgb_to_linear(c):
    """Convert sRGB [0,1] to linear RGB (IEC 61966-2-1)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Convert linear RGB to sRGB [0,1]."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


# ============================================================
# Vienot 1999 deuteranopia simulation
# ============================================================

DEUTAN_MATRIX = np.array([
    [0.29275, 0.70725, 0.00000],
    [0.29275, 0.70725, 0.00000],
    [-0.02234, 0.02234, 1.00000],
])


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB array [0, 1]."""
    h = hex_color.lstrip('#')
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])


def rgb_to_hex(rgb):
    """Convert RGB array [0, 1] to hex string."""
    r, g, b = (np.clip(rgb, 0, 1) * 255).astype(int)
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)


def simulate_deuteranopia(hex_color):
    """Simulate how a deuteranope sees a hex color (Vienot 1999)."""
    rgb = hex_to_rgb(hex_color)
    linear = srgb_to_linear(rgb)
    simulated = np.clip(DEUTAN_MATRIX @ linear, 0, 1)
    srgb_out = np.clip(linear_to_srgb(simulated), 0, 1)
    return rgb_to_hex(srgb_out)


# ============================================================
# RGB to CIELAB via XYZ (D65 illuminant)
# ============================================================

# sRGB to XYZ matrix (D65, IEC 61966-2-1)
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# D65 reference white
D65_WHITE = np.array([0.95047, 1.00000, 1.08883])


def rgb_to_lab(hex_color):
    """Convert hex color to CIELAB via linearized sRGB -> XYZ -> Lab."""
    rgb = hex_to_rgb(hex_color)
    linear = srgb_to_linear(rgb)
    xyz = SRGB_TO_XYZ @ linear
    xyz_n = xyz / D65_WHITE

    # CIE f function
    f = np.where(
        xyz_n > 0.008856,
        xyz_n ** (1.0 / 3.0),
        7.787 * xyz_n + 16.0 / 116.0
    )

    L = 116.0 * f[1] - 16.0
    a = 500.0 * (f[0] - f[1])
    b = 200.0 * (f[1] - f[2])
    return L, a, b


# ============================================================
# CIEDE2000 delta E (Sharma, Wu, Dalal 2005)
# ============================================================

def ciede2000(lab1, lab2):
    """
    Compute CIEDE2000 color difference between two CIELAB colors.

    Reference: Sharma, Wu, Dalal (2005)
    "The CIEDE2000 Color-Difference Formula: Implementation Notes,
    Supplementary Test Data, and Mathematical Observations"
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: Calculate C'ab, h'ab
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    C_avg_7 = C_avg**7
    G = 0.5 * (1.0 - np.sqrt(C_avg_7 / (C_avg_7 + 25.0**7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    # Step 2: Calculate delta L', delta C', delta H'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        delta_h_prime = 0.0
    elif abs(h2_prime - h1_prime) <= 180:
        delta_h_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > 180:
        delta_h_prime = h2_prime - h1_prime - 360
    else:
        delta_h_prime = h2_prime - h1_prime + 360

    delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

    # Step 3: Calculate CIEDE2000
    L_avg_prime = (L1 + L2) / 2.0
    C_avg_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime == 0:
        h_avg_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180:
        h_avg_prime = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360:
        h_avg_prime = (h1_prime + h2_prime + 360) / 2.0
    else:
        h_avg_prime = (h1_prime + h2_prime - 360) / 2.0

    T = (1.0
         - 0.17 * np.cos(np.radians(h_avg_prime - 30))
         + 0.24 * np.cos(np.radians(2 * h_avg_prime))
         + 0.32 * np.cos(np.radians(3 * h_avg_prime + 6))
         - 0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))

    S_L = 1.0 + 0.015 * (L_avg_prime - 50)**2 / np.sqrt(20 + (L_avg_prime - 50)**2)
    S_C = 1.0 + 0.045 * C_avg_prime
    S_H = 1.0 + 0.015 * C_avg_prime * T

    C_avg_prime_7 = C_avg_prime**7
    R_C = 2.0 * np.sqrt(C_avg_prime_7 / (C_avg_prime_7 + 25.0**7))
    delta_theta = 30.0 * np.exp(-((h_avg_prime - 275) / 25.0)**2)
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    # Parametric weighting factors (all 1.0 for standard usage)
    k_L = k_C = k_H = 1.0

    delta_E = np.sqrt(
        (delta_L_prime / (k_L * S_L))**2
        + (delta_C_prime / (k_C * S_C))**2
        + (delta_H_prime / (k_H * S_H))**2
        + R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )

    return delta_E


# ============================================================
# Main analysis
# ============================================================

def analyze_color_group(name, colors):
    """Analyze a color group: simulate deuteranopia and compute pairwise delta E."""
    print(f"\n### {name}\n")

    # --- Per-color simulation table ---
    print("#### Deuteranopia Simulation\n")
    print("| Color Name | Original Hex | Simulated Hex | L* | a* | b* |")
    print("|------------|-------------|---------------|------|------|------|")

    simulated = {}
    labs = {}
    for color_name, hex_val in colors.items():
        sim_hex = simulate_deuteranopia(hex_val)
        L, a, b = rgb_to_lab(sim_hex)
        simulated[color_name] = sim_hex
        labs[color_name] = (L, a, b)
        print(f"| {color_name} | {hex_val} | {sim_hex} | {L:.1f} | {a:.1f} | {b:.1f} |")

    # --- Pairwise delta E table ---
    color_names = list(colors.keys())
    pairs = list(combinations(color_names, 2))

    if not pairs:
        print("\n*Only one color -- no pairwise comparison needed.*\n")
        return [], simulated, labs

    print(f"\n#### Pairwise Delta E ({len(pairs)} pairs)\n")
    print("| Color A | Color B | Delta E | Confusable |")
    print("|---------|---------|---------|------------|")

    results = []
    for a_name, b_name in pairs:
        de = ciede2000(labs[a_name], labs[b_name])
        confusable = "Yes" if de < DELTA_E_THRESHOLD else "No"
        results.append((a_name, b_name, de, confusable))

    # Sort by delta E ascending
    results.sort(key=lambda x: x[2])

    for a_name, b_name, de, confusable in results:
        print(f"| {a_name} | {b_name} | {de:.2f} | {confusable} |")

    return results, simulated, labs


def main():
    print("# Colorblind Accessibility Findings (Deuteranopia)")
    print()
    print("**Date:** 2026-03-27")
    print("**Methodology:** Vienot, Brettel & Mollon 1999 deuteranopia simulation matrix")
    print("+ CIEDE2000 delta E (Sharma, Wu, Dalal 2005)")
    print(f"**Confusability threshold:** delta E < {DELTA_E_THRESHOLD}")
    print()
    print("## Overview")
    print()
    print("This document reports the results of algorithmic deuteranopia simulation")
    print("on the VHP4Safety color palette used in the AOP-Wiki RDF Dashboard.")
    print("Deuteranopia (green-blind) is the most common form of color vision deficiency,")
    print("affecting approximately 1% of males.")
    print()
    print("Colors that appear distinct to trichromats may become confusable under")
    print("deuteranopia. Pairs with CIEDE2000 delta E < 10 are flagged as potentially")
    print("confusable in chart contexts (small markers, thin lines, adjacent legend items).")

    # Analyze all three color groups
    palette_results, palette_sim, palette_labs = analyze_color_group(
        "VHP4Safety Palette (11 colors)", PALETTE_COLORS
    )
    oecd_results, oecd_sim, oecd_labs = analyze_color_group(
        "OECD Status Colors (8 colors)", OECD_STATUS_COLORS
    )
    type_results, type_sim, type_labs = analyze_color_group(
        "Property Type Colors (6 colors)", TYPE_COLORS
    )

    # --- Confusable Pairs Summary ---
    print("\n## Confusable Pairs Summary\n")
    print("Pairs where delta E < 10 under deuteranopia simulation:\n")

    any_confusable = False

    for group_name, results in [
        ("Palette", palette_results),
        ("OECD Status", oecd_results),
        ("Type Colors", type_results),
    ]:
        confusable = [(a, b, de) for a, b, de, c in results if c == "Yes"]
        if confusable:
            any_confusable = True
            print(f"### {group_name}\n")
            print("| Color A | Color B | Delta E |")
            print("|---------|---------|---------|")
            for a, b, de in confusable:
                print(f"| {a} | {b} | {de:.2f} |")
            print()

    if not any_confusable:
        print("No confusable pairs found across any color group.\n")

    # --- Affected Plots Preview ---
    print("\n## Affected Color Groups\n")
    print("The following color groups contain confusable pairs and are used in dashboard plots:")
    print()

    for group_name, results in [
        ("Palette (`BRAND_COLORS['palette']`)", palette_results),
        ("OECD Status (`BRAND_COLORS['oecd_status']`)", oecd_results),
        ("Type Colors (`BRAND_COLORS['type_colors']`)", type_results),
    ]:
        confusable_count = sum(1 for _, _, _, c in results if c == "Yes")
        if confusable_count > 0:
            print(f"- **{group_name}**: {confusable_count} confusable pair(s)")
        else:
            print(f"- **{group_name}**: No confusable pairs")

    print()
    print("Specific per-plot impact analysis is deferred to Plan 02 (full audit report),")
    print("which will cross-reference each plot's color usage against these confusable pairs.")

    # --- Methodology Note ---
    print("\n## Methodology\n")
    print("### Deuteranopia Simulation")
    print("- **Model:** Vienot, Brettel & Mollon (1999) \"Digital Video Colourmaps for")
    print("  Checking the Legibility of Displays by Dichromats\"")
    print("- **Matrix:** 3x3 linear transformation in linearized sRGB space")
    print("- **Linearization:** IEC 61966-2-1 sRGB transfer function (piecewise: threshold 0.04045)")
    print()
    print("### Color Difference Metric")
    print("- **Formula:** CIEDE2000 (Sharma, Wu, Dalal 2005)")
    print("- **Color space:** CIELAB via XYZ with D65 illuminant")
    print(f"- **Threshold:** delta E < {DELTA_E_THRESHOLD} flags a pair as confusable")
    print()
    print("### Scope")
    print("- Only deuteranopia (the most common CVD type) was evaluated, per decision D-09.")
    print("- Protanopia and tritanopia were not assessed.")
    print("- No fix or remediation suggestions are included, per decision D-11.")
    print("  Phase 8 will determine remediation based on these findings.")


if __name__ == "__main__":
    main()
