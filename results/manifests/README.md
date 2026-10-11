# results/manifests/

Small `manifest.json` files of runs that the paper cites, one subdirectory per
run (`<tool>/<config>[/<seed>]/manifest.json`). Heavy outputs stay on Drive.
`scripts/make_figures.py` and `scripts/make_tables.py` read ONLY these, and
refuse incomplete, smoke, or `-dirty` manifests.
