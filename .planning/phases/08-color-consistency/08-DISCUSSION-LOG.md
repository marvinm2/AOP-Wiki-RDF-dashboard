# Phase 8: Color Consistency - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 08-color-consistency
**Areas discussed:** FIX NOW scope, Color framework doc, Fix approach, Verification

---

## FIX NOW Scope

| Option | Description | Selected |
|--------|-------------|----------|
| FIX NOW only (19 plots) | Focus on 19 color-critical plots. FIX LATER have non-color issues out of scope. | |
| FIX NOW + FIX LATER (23 plots) | Include the 4 FIX LATER plots too. Their color decisions are documented. | ✓ |
| FIX NOW + colorblind fixes | 19 FIX NOW plus address 2 colorblind-confusable pairs. | |

**User's choice:** FIX NOW + FIX LATER (23 plots)
**Notes:** None

### Wiring Defects

| Option | Description | Selected |
|--------|-------------|----------|
| Fix wiring too | 3 defects are small, natural to fix while touching those files | ✓ |
| Color only, skip wiring | Keep Phase 8 purely about color | |

**User's choice:** Fix wiring too
**Notes:** None

### Colorblind Pairs

| Option | Description | Selected |
|--------|-------------|----------|
| Defer colorblind fixes | Document only. Pairs occur in non-FIX NOW plots, palette is frozen. | |
| Fix if safe | Address only if fix doesn't require modifying BRAND_COLORS['palette'] | ✓ |
| Fix all colorblind issues | Replace confusable pairs everywhere, even with plot-specific overrides | |

**User's choice:** Fix if safe
**Notes:** None

---

## Color Framework Doc

### Location

| Option | Description | Selected |
|--------|-------------|----------|
| Extend .claude/colors.md | Add section to existing file. All color guidance in one place. | ✓ |
| New .claude/color-framework.md | Separate file for decision rules. | |
| In CLAUDE.md | Short section always visible to agents. | |

**User's choice:** Extend .claude/colors.md
**Notes:** None

### Detail Level

| Option | Description | Selected |
|--------|-------------|----------|
| Decision tree only | Simple flowchart rule + quick-reference table. ~10-15 lines. | ✓ |
| Comprehensive guide | Decision tree + rationale + examples + colorblind notes. ~50 lines. | |
| Minimal one-liner | 2-3 lines in colors.md. | |

**User's choice:** Decision tree only
**Notes:** None

---

## Fix Approach

### Legacy Aliases

| Option | Description | Selected |
|--------|-------------|----------|
| Replace aliases with hex | Change secondary → magenta, accent → blue, etc. Clearer intent. | ✓ |
| Leave aliases, fix color only | Only change plots where audit says color is wrong. | |
| Remove legacy aliases entirely | Replace usages AND delete keys from BRAND_COLORS. | |

**User's choice:** Replace aliases with explicit named keys
**Notes:** None

### Color Reference Style

| Option | Description | Selected |
|--------|-------------|----------|
| BRAND_COLORS['blue'] | References dict for maintainability. | ✓ |
| Direct hex '#307BBF' | Explicit and self-documenting. | |
| You decide | Claude's discretion. | |

**User's choice:** BRAND_COLORS['blue']
**Notes:** None

### Pie Chart Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Fix colors on current chart type | Apply audit's color decision to pies as-is. Chart type conversion is v2. | ✓ |
| Skip pie chart plots entirely | Don't touch chart-type-flagged plots. | |
| Convert pies to bars too | Fix both chart type and color. Adds scope. | |

**User's choice:** Fix colors on current chart type
**Notes:** None

---

## Verification

### Method

| Option | Description | Selected |
|--------|-------------|----------|
| Automated grep + checklist | Grep for antipatterns + manual checklist matching audit entries to fixes. | ✓ |
| Grep only | Automated scan only. Fast but may miss subtle issues. | |
| Visual spot-check | Run dashboard and visually inspect. Thorough but needs running instance. | |

**User's choice:** Automated grep + checklist
**Notes:** None

### Regression Check

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, regression check | Verify SKIP plots weren't accidentally modified via git diff. | ✓ |
| No, trust the process | Only verify changed plots. | |

**User's choice:** Yes, regression check
**Notes:** None

---

## Claude's Discretion

- Order of plot fixes (by module, priority, or fix similarity)
- Exact grep patterns for antipattern detection
- Commit granularity (batch similar fixes vs one per plot)

## Deferred Ideas

- Chart type conversions (pie → bar for 5 plots) — v2
- Removal of legacy alias keys from BRAND_COLORS — only replacing usages
- Plot title rewrites — v2
