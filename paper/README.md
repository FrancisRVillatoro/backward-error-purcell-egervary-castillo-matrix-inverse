# Manuscript sources

This directory contains the definitive manuscript sources corresponding to
release 1.0.0.

Files:

- `main_simax.tex` — main manuscript;
- `supplement_simax.tex` — supplementary materials, including SM12 on the
  reproducibility archive and retained bulk outputs;
- `castillo_refs.bib` — bibliography database;
- `siamart251216.cls` — SIAM article class used for compilation;
- `siamplain.bst` — SIAM bibliography style.

Compile the main manuscript with:

    pdflatex main_simax.tex
    bibtex main_simax
    pdflatex main_simax.tex
    pdflatex main_simax.tex

Compile the supplementary materials with:

    pdflatex supplement_simax.tex
    bibtex supplement_simax
    pdflatex supplement_simax.tex
    pdflatex supplement_simax.tex

The journal-published article, once available, should be treated as the
version of record.
