# VHP4Safety Brand Colors Reference

## Application Color Scheme
- **Headers & Footers**: `#29235C` (Primary Dark)
- **Body Background**: `#f5f5f5` (Light Gray)
- **Content Cards**: `white` with subtle shadows
- **Landing Hero**: `#29235C` (Primary Dark)

## Primary Colors
- **Primary Dark**: `#29235C` (C100 M100 Y25 K25)
- **Primary Magenta**: `#E6007E` (C0 M100 Y50 K0)
- **Primary Blue**: `#307BBF` (C80 M45 Y0 K0)

## Secondary Colors
- **Light Blue**: `#009FE3` (C100 M0 Y20 K0)
- **Orange**: `#EB5B25` (C0 M75 Y90 K0)
- **Sky Blue**: `#93D5F6` (C45 M0 Y100 K0)
- **Deep Magenta**: `#9A1C57` (C40 M100 Y40 K10)
- **Teal**: `#45A6B2` (C70 M15 Y30 K0)
- **Purple**: `#B81178` (C30 M100 Y10 K0)
- **Dark Teal**: `#005A6C` (C90 M45 Y40 K30)
- **Violet**: `#64358C` (C75 M90 Y0 K0)

## Usage Guidelines
- Headers/footers: `#29235C` across all pages
- Main backgrounds: `#f5f5f5` light gray
- Navigation: `#307BBF` (Primary Blue)
- CTAs/buttons: `#E6007E` (Primary Magenta)

## Color Decision Framework

### Decision Tree

1. **Is the data categorical?** (distinct named groups: entity types, ontology sources, OECD statuses)
   - YES: Use `BRAND_COLORS['palette']` or semantic mapping (`type_colors`, `oecd_status`)
   - NO: Continue to step 2

2. **Is it a single metric or series?** (one count, one average, one density)
   - YES: Use `BRAND_COLORS['blue']` (#307BBF)
   - NO: Continue to step 3

3. **Are there multiple views of the same entity?** (absolute + delta, created + modified)
   - YES: Use `BRAND_COLORS['blue']` for ALL outputs (consistency)
   - NO: Consult team

### Quick Reference

| Plot Type | Color Treatment | Example |
|-----------|----------------|---------|
| Single-metric bar | `marker_color=BRAND_COLORS['blue']` | Entity counts, reuse distribution |
| Single-metric line | `color_discrete_sequence=[BRAND_COLORS['blue']]` | Author counts, network density |
| Multi-series line (same concept) | `BRAND_COLORS['blue']` for all | AOP lifetime views |
| Categorical bar/line | `color_discrete_sequence=BRAND_COLORS['palette']` | KE components, avg per AOP |
| Semantic categories | `color_discrete_map=BRAND_COLORS['type_colors']` or `oecd_status` | Completeness by status |
| Boxplot | `marker=dict(color=BRAND_COLORS['primary'])` | Completeness distribution |

### Legacy Aliases (do not use in new code)

| Legacy Alias | Replacement |
|--------------|-------------|
| `BRAND_COLORS['secondary']` | `BRAND_COLORS['magenta']` |
| `BRAND_COLORS['accent']` | `BRAND_COLORS['blue']` |
| `BRAND_COLORS['light']` | `BRAND_COLORS['sky_blue']` |
