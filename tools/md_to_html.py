#!/usr/bin/env python3
"""Convert paper/PAPER.md to a self-contained HTML (for Chrome print-to-PDF).

Usage: python tools/md_to_html.py
Requires: pip install markdown
"""
from __future__ import annotations
import os
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "paper", "PAPER.md")
DST = os.path.join(ROOT, "paper", "PAPER.html")

CSS = """
:root { --ink:#1a1a2e; --muted:#6b7280; --blue:#2563eb; --green:#16a34a; --red:#dc2626; }
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-size: 10.5pt; }
body { font-family: "Charter", "Georgia", "Times New Roman", serif; color: var(--ink);
       line-height: 1.5; max-width: 100%; margin: 0; }
h1 { font-size: 20pt; line-height: 1.25; margin: 0 0 6pt 0; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #e5e7eb; padding-bottom: 3pt;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p, li { text-align: justify; }
code { font-family: "SF Mono", "Menlo", monospace; font-size: 8.8pt; background: #f3f4f6;
       padding: 0.5pt 2.5pt; border-radius: 3px; }
pre { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 4px; padding: 7pt 9pt;
      font-size: 8.6pt; overflow-x: hidden; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.8pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
img { max-width: 100%; display: block; margin: 10pt auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid var(--blue); margin: 8pt 0; padding: 2pt 10pt;
             color: #374151; background: #f8fafc; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 14pt 0; }
a { color: var(--blue); text-decoration: none; }
strong { color: #111827; }
.cover { text-align: center; margin-bottom: 14pt; }
.cover .meta { color: var(--muted); font-size: 9pt; }
.cover hr { width: 40%; margin: 10pt auto; }
"""

COVER = """
<div class="cover">
  <div class="meta">UCM — Universal Control Model · technical report v1.0 · 2026-09-25 · private pre-publication</div>
  <hr/>
</div>
"""


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        text = f.read()
    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc", "md_in_html"],
    )
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>UCM — Technical Report v1.0</title>
<style>{CSS}</style>
</head>
<body>
{COVER}
{html}
</body>
</html>
"""
    with open(DST, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", DST)


if __name__ == "__main__":
    main()
