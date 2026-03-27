#!/usr/bin/env python3
"""
Deuteranopia simulation and pairwise distinguishability analysis
for the VHP4Safety palette.

Methodology:
  - Vienot, Brettel & Mollon 1999 deuteranopia simulation matrix
  - sRGB linearization (IEC 61966-2-1)
  - RGB to CIELAB via XYZ (D65 illuminant)
  - CIEDE2000 delta E (Sharma, Wu, Dalal 2005)
  - Confusability threshold: delta E < 10

Usage:
  python colorblind_analysis.py
"""

import numpy as np
from itertools import combinations

# ---------------------------------------------------------------------------
# VHP4Safety palette colors (from plots/shared.py BRAND_COLORS)
# ---------------------------------------------------------------------------

PALETTE_COLORS = {
    'primary':       '#29235C',
    'magenta':       '#E6007E',
    'blue':          '#307BBF',
    'light_blue':    '#009FE3',
    'orange':        '#EB5B25',
    'sky_blue':      '#93D5F6',
    'deep_magenta':  '#9A1C57',
    'teal':          '#45A6B2',
    'warm_pink':     '#B81178',
    'dark_teal':     '#005A6C',
    'violet':        '#64358C',
}

OECD_STATUS_COLORS = {
    'EAGMST Under Review':       '#307BBF',
    'Under Development':         '#009FE3',
    'TFHA/WNT Endorsed':         '#29235C',
    'WNT Endorsed':              '#E6007E',
    'Approved':                  '#EB5B25',
    'No Status':                 '#999999',
    'EAGMST Under Development':  '#45A6B2',
    'Not OECD':                  '#93D5F6',
}

TYPE_COLORS = {
    'Essential':  '#29235C',
    'Metadata':   '#E6007E',
    'Content':    '#EB5B25',
    'Context':    '#93D5F6',
    'Assessment': '#307BBF',
    'Structure':  '#45A6B2',
}

DELTA_E_THRESHOLD = 10

# ---------------------------------------------------------------------------
# sRGB linearization / delinearization
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    """Convert sRGB [0,1] to linear RGB (vectorized)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    """Convert linear RGB to sRGB [0,1] (vectorized)."""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

# ---------------------------------------------------------------------------
# Vienot 1999 deuteranopia simulation
# ---------------------------------------------------------------------------

DEUTAN_MATRIX = np.array([
    [0.29275,  0.70725,  0.00000],
    [0.29275,  0.70725,  0.00000],
    [-0.02234, 0.02234,  1.00000],
])

def hex_to_rgb(hex_color):
    """Convert '#RRGGBB' to numpy array [0,1]."""
    h = hex_color.lstrip('#')
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])

def rgb_to_hex(rgb):
    """Convert numpy array [0,1] to '#rrggbb'."""
    c = np.clip(np.round(rgb * 255), 0, 255).astype(int)
    return '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])

def simulate_deuteranopia(hex_color):
    """Simulate how a deuteranope perceives a hex color.

    Pipeline: hex -> sRGB -> linearize -> matrix -> clip -> inverse sRGB -> clip -> hex
    """
    srgb = hex_to_rgb(hex_color)
    linear = srgb_to_linear(srgb)
    simulated_linear = np.clip(DEUTAN_MATRIX @ linear, 0, 1)
    simulated_srgb = np.clip(linear_to_srgb(simulated_linear), 0, 1)
    return rgb_to_hex(simulated_srgb)

# ---------------------------------------------------------------------------
# RGB to CIELAB (via XYZ, D65 illuminant)
# ---------------------------------------------------------------------------

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

D65_WHITE = np.array([0.95047, 1.00000, 1.08883])

def _cie_f(t):
    """CIE Lab f function."""
    return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

def rgb_to_lab(hex_color):
    """Convert a hex color to CIELAB (L*, a*, b*) via linearized sRGB -> XYZ -> Lab."""
    srgb = hex_to_rgb(hex_color)
    linear = srgb_to_linear(srgb)
    xyz = SRGB_TO_XYZ @ linear
    xyz_n = xyz / D65_WHITE
    f = _cie_f(xyz_n)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return L, a, b

# ---------------------------------------------------------------------------
# CIEDE2000 delta E  (Sharma, Wu, Dalal 2005)
# ---------------------------------------------------------------------------

def ciede2000(lab1, lab2):
    """Compute CIEDE2000 color difference between two CIELAB colors.

    Reference: Sharma, Wu, Dalal (2005) "The CIEDE2000 Color-Difference Formula:
    Implementation Notes, Supplementary Test Data, and Mathematical Observations"
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: compute C'ab and h'ab
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    C_avg_7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C_avg_7 / (C_avg_7 + 25**7)))

    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    # Step 2: compute delta L', delta C', delta H'
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        dh_prime = 0
    elif abs(h2_prime - h1_prime) <= 180:
        dh_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > 180:
        dh_prime = h2_prime - h1_prime - 360
    else:
        dh_prime = h2_prime - h1_prime + 360

    dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))

    # Step 3: compute CIEDE2000
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

    T = (1
         - 0.17 * np.cos(np.radians(h_avg_prime - 30))
         + 0.24 * np.cos(np.radians(2 * h_avg_prime))
         + 0.32 * np.cos(np.radians(3 * h_avg_prime + 6))
         - 0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))

    SL = 1 + 0.015 * (L_avg_prime - 50)**2 / np.sqrt(20 + (L_avg_prime - 50)**2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T

    C_avg_prime_7 = C_avg_prime ** 7
    RC = 2 * np.sqrt(C_avg_prime_7 / (C_avg_prime_7 + 25**7))
    d_theta = 30 * np.exp(-((h_avg_prime - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    dE = np.sqrt(
        (dL_prime / SL)**2
        + (dC_prime / SC)**2
        + (dH_prime / SH)**2
        + RT * (dC_prime / SC) * (dH_prime / SH)
    )
    return dE

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyze_color_group(name, colors):
    """Simulate deuteranopia and compute pairwise delta E for a color group.

    Returns (sim_table, pair_table, confusable_pairs).
    """
    # Simulation table
    sim_rows = []
    sim_hex = {}
    sim_lab = {}
    for cname, chex in colors.items():
        dhex = simulate_deuteranopia(chex)
        L, a, b = rgb_to_lab(dhex)
        sim_rows.append((cname, chex, dhex, L, a, b))
        sim_hex[cname] = dhex
        sim_lab[cname] = (L, a, b)

    # Pairwise delta E
    pair_rows = []
    confusable = []
    names = list(colors.keys())
    for n1, n2 in combinations(names, 2):
        de = ciede2000(sim_lab[n1], sim_lab[n2])
        is_confusable = de < DELTA_E_THRESHOLD
        pair_rows.append((n1, n2, de, is_confusable))
        if is_confusable:
            confusable.append((n1, n2, de))

    pair_rows.sort(key=lambda r: r[2])
    confusable.sort(key=lambda r: r[2])
    return sim_rows, pair_rows, confusable

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_simulation_table(title, sim_rows):
    print(f"\n### {title}\n")
    print("| Color Name | Original Hex | Simulated Hex | L* | a* | b* |")
    print("|------------|-------------|---------------|------|------|------|")
    for name, orig, sim, L, a, b in sim_rows:
        print(f"| {name} | {orig} | {sim} | {L:.1f} | {a:.1f} | {b:.1f} |")

def print_pair_table(title, pair_rows):
    print(f"\n### {title}\n")
    print("| Color A | Color B | Delta E | Confusable |")
    print("|---------|---------|---------|------------|")
    for n1, n2, de, conf in pair_rows:
        flag = "Yes" if conf else "No"
        print(f"| {n1} | {n2} | {de:.2f} | {flag} |")

def print_confusable_summary(title, confusable, group_label):
    print(f"\n### {title}\n")
    if not confusable:
        print("No confusable pairs found (all pairs delta E >= 10).\n")
        return
    print(f"The following pairs from **{group_label}** have delta E < {DELTA_E_THRESHOLD} under deuteranopia simulation:\n")
    print("| Color A | Color B | Delta E | Group |")
    print("|---------|---------|---------|-------|")
    for n1, n2, de in confusable:
        print(f"| {n1} | {n2} | {de:.2f} | {group_label} |")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("# Deuteranopia Simulation Results\n")
    print("**Method:** Vienot, Brettel & Mollon 1999 deuteranopia matrix + CIEDE2000 delta E")
    print(f"**Confusability threshold:** delta E < {DELTA_E_THRESHOLD}")
    print(f"**Colors analyzed:** {len(PALETTE_COLORS)} palette + {len(OECD_STATUS_COLORS)} OECD status + {len(TYPE_COLORS)} type colors\n")

    # --- Palette ---
    print("## 1. VHP4Safety Palette (11 colors)\n")
    sim, pairs, conf = analyze_color_group("Palette", PALETTE_COLORS)
    print_simulation_table("Palette Simulation", sim)
    print_pair_table("Palette Pairwise Delta E (sorted ascending)", pairs)
    print_confusable_summary("Palette Confusable Pairs (delta E < 10)", conf, "palette")

    # --- OECD Status ---
    print("\n## 2. OECD Status Colors (8 colors)\n")
    sim2, pairs2, conf2 = analyze_color_group("OECD Status", OECD_STATUS_COLORS)
    print_simulation_table("OECD Status Simulation", sim2)
    print_pair_table("OECD Status Pairwise Delta E (sorted ascending)", pairs2)
    print_confusable_summary("OECD Status Confusable Pairs (delta E < 10)", conf2, "oecd_status")

    # --- Type Colors ---
    print("\n## 3. Type Colors (6 colors)\n")
    sim3, pairs3, conf3 = analyze_color_group("Type Colors", TYPE_COLORS)
    print_simulation_table("Type Colors Simulation", sim3)
    print_pair_table("Type Colors Pairwise Delta E (sorted ascending)", pairs3)
    print_confusable_summary("Type Colors Confusable Pairs (delta E < 10)", conf3, "type_colors")

    # --- Cross-group summary ---
    print("\n## 4. Overall Confusable Pairs Summary\n")
    all_conf = []
    for c, label in [(conf, "palette"), (conf2, "oecd_status"), (conf3, "type_colors")]:
        for n1, n2, de in c:
            all_conf.append((n1, n2, de, label))
    all_conf.sort(key=lambda r: r[2])

    if not all_conf:
        print("No confusable pairs found across any color group.\n")
    else:
        print(f"Total confusable pairs (delta E < {DELTA_E_THRESHOLD}): **{len(all_conf)}**\n")
        print("| Color A | Color B | Delta E | Group |")
        print("|---------|---------|---------|-------|")
        for n1, n2, de, grp in all_conf:
            print(f"| {n1} | {n2} | {de:.2f} | {grp} |")

    print("\n## 5. Affected Color Groups\n")
    affected = set()
    if conf:
        affected.add("palette")
    if conf2:
        affected.add("oecd_status")
    if conf3:
        affected.add("type_colors")
    if affected:
        for g in sorted(affected):
            print(f"- **{g}** contains confusable pairs under deuteranopia")
    else:
        print("No color groups contain confusable pairs.")

    print("\n---")
    print("*Generated by colorblind_analysis.py*")


if __name__ == "__main__":
    main()
