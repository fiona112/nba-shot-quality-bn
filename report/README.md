# Project report

`main.tex` revises the team's supplied LaTeX draft using the verified repository
results. `main.pdf` is the compiled review copy. Individual contributions and
reflections await team completion. Anna's EM and player analyses from merged PR
#4 are integrated with checked input cohorts and reproduced results. The selected
outer-training EM fit reaches its fixed iteration cap without meeting tolerance;
the report discloses this limitation. The report is a draft for team review, not
a submission.

The current main text (Sections 1–6) is within the seven-page course limit.
References and appendices follow separately. Check the limit again after adding
individual statements or verifying extensions.

## Build

From the repository root, using the Python environment in the root README:

```bash
python report/build_tables.py
cd report
tectonic --untrusted main.tex
```

The reviewed PDF was built with Tectonic 0.17.0. Its first run downloads TeX
packages. A standard LaTeX installation can instead run `pdflatex main.tex`
twice from this directory. No bibliography download or raw NBA data is needed:
the generated tables are committed, and the table builder reads committed
experiment results plus the embedded preprocessing metadata.

For Overleaf, upload the `report/` folder and `results/figures/` while preserving
their relative paths, and select `report/main.tex` as the main document. The
source uses the official NeurIPS 2026 preprint style in `neurips_2026.sty`,
downloaded unchanged from the [conference formatting archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
The style's attribution header is preserved. The team draft supplied the title,
authors, overall report structure, and the initial extension descriptions.

`build_tables.py` checks result/figure hashes against experiment metadata before
regenerating `generated/*.tex`. `generated/input_manifest.json` records the exact
source and result inputs. Hash mismatch means a result was replaced without its
matching provenance; regenerate the relevant experiment before rebuilding.
