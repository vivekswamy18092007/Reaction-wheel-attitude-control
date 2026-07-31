# report/

Report source (LaTeX/Markdown/figures) goes here. Nothing generated
automatically lands in this folder -- `deimos run`/`compare`/`tune` write to
`runs/<timestamp>_<name>/` instead; pull the specific figures/tables you want
to cite into a report draft here by hand.

`legacy/PD_controller_valuetest.py` was the old ad hoc report-figure
generator (Scenario A gain design + 4-plot report figure) -- a candidate to
port into a real `deimos`-based script here once someone gets to it.
