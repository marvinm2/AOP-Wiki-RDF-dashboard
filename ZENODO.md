# Zenodo archiving & citation

Three Zenodo records make this resource citable. Two are **new** and need a
one-time activation (the steps below); one already exists.

| Record | Type | DOI | Status |
|---|---|---|---|
| AOP-Wiki RDF **dataset** | data | `10.5281/zenodo.13353286` | live (auto, monthly) |
| AOP-Wiki RDF Dashboard **software** | software | TODO | activate (WS4) |
| Multi-endpoint pipeline **software** | software | TODO | activate (WS4) |
| Per-version **plot exports** | data | TODO | activate (WS5) |

## WS4 — software DOIs (GitHub → Zenodo, one-time)

Each repo carries `.zenodo.json` + `CITATION.cff`; GitHub's native Zenodo
integration mints a software DOI on each GitHub *release*.

For **both** `marvinm2/AOP-Wiki-RDF-dashboard` and
`marvinm2/AOP-Wiki_multi-endpoint`:

1. Sign in at https://zenodo.org with GitHub; under **Account → GitHub**, flip the
   repo's toggle **On** (this installs the release webhook).
2. On GitHub, **Create a new release** (e.g. tag `v1.0.0`). Zenodo captures the
   tarball and reads `.zenodo.json` for the metadata.
3. Copy the minted **concept DOI** badge into the repo README's citation section.
4. (Optional) add the concept DOI to `CITATION.cff` `identifiers:`.

No further automation needed — every future release auto-archives.

## WS5 — per-version plot-data record (scripted)

A dedicated record holds the per-quarter plot bundles produced by
`scripts/export_all_versions.py` (PNG + SVG + CSV + manifest, plus a trends
bundle). `.github/workflows/upload-plots-zenodo.yml` publishes a new version of
it on manual dispatch.

One-time activation:

1. Create a new **empty** Zenodo upload (Type: Dataset), save the draft, and note
   its **deposition ID** (the number in the URL). Don't publish it by hand.
2. Create a Zenodo **personal access token** with `deposit:write` +
   `deposit:actions`.
3. Add two repo secrets to `AOP-Wiki-RDF-dashboard`:
   - `ZENODO_ACCESS_TOKEN` — the token (can be the same one the dataset record uses)
   - `ZENODO_PLOTS_DEPOSITION_ID` — the deposition ID from step 1
4. Run the **upload-plots-zenodo** workflow (Actions tab → Run workflow). It
   generates the latest bundle + trends and publishes a new version. Until the
   secrets exist the workflow exits cleanly as a no-op.

Per quarter: after loading the new version onto the endpoint, run the workflow
again to publish a fresh plot-data version (typically as a step after the
quarterly-update PR is merged and deployed).

### Local dry-run of the export (no Zenodo)

```bash
cd AOP-Wiki-RDF-dashboard
python scripts/export_all_versions.py --out exports --zip          # latest + trends
python scripts/export_all_versions.py 2026-04-01 --out exports     # a specific version
python scripts/export_all_versions.py --all --out exports --zip    # every version (slow)
```
