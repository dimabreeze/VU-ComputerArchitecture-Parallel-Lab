#!/usr/bin/env python3
"""Builds the report (PDF) from Markdown parts.

Usage: scripts/build_report.py [part ...]      e.g. 0_ivadas 1_openmp (default: all)
Input:  reports/ataskaita/src/<part>.md, Markdown with a metadata header:
          title: ...        part title (title page)
          subtitle: ...     optional
          part: ...         e.g. "Bendroji dalis" / "1 dalis" (title page, page footer)
        Images are referenced from the project root: ![caption](docs/img/load/seq-base.png);
        they become numbered figures ("1 pav. caption"), always one under another (side-by-side
        pairs made the text in plots and screenshots unreadable).
        Call-out boxes (Markdown "admonition" syntax):
          !!! fact "Svarbiausi faktai"        key findings (accent colour)
          !!! conditions "Matavimo sąlygos"   measurement conditions (amber)
          !!! note "Pastaba"                  side note (grey)
Output: reports/ataskaita/<part>.pdf for every part, and reports/ataskaita/ataskaita_visa.pdf
        (all parts in order). PDFs are printed by headless Google Chrome.
Everything is built in a temporary directory and moved in at the end (OneDrive).
"""
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports/ataskaita/src"
OUT = ROOT / "reports/ataskaita"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
AUTHOR = "Dmitrij Kosmakov"
COURSE = "Kompiuterių architektūra · 4 laboratorinis darbas · Vilniaus universitetas"
DATE = "2026 m. lapkritis"
REPO = "https://github.com/dimabreeze/VU-ComputerArchitecture-Parallel-Lab"
QR = ROOT / "docs/img/qr-github.png"  # QR code of REPO (title pages)

CSS = r"""
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
  @bottom-left { content: var(--footer); font: 8pt "Avenir Next", sans-serif; color: #8a96a8; }
  @bottom-right { content: counter(page); font: 8pt "Avenir Next", sans-serif; color: #8a96a8; } }
@page :first { @bottom-left { content: none; } @bottom-right { content: none; } }
:root { --accent: #2f6690; --ink: #172033; --muted: #5b6474; --line: #d9dee5; }
html { font: 10.5pt/1.5 "Charter", "Georgia", serif; color: var(--ink); }
body { margin: 0; }
h1, h2, h3, h4, .title, .meta, figcaption, table, .admonition-title { font-family: "Avenir Next", "Helvetica Neue", sans-serif; }
h1 { font-size: 20pt; margin: 0 0 6mm; color: var(--ink); break-before: page; }
h2 { font-size: 14pt; margin: 8mm 0 3mm; color: var(--accent); break-after: avoid; }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; break-after: avoid; }
p { margin: 0 0 3mm; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
code { font: 9pt "Menlo", monospace; background: #f1f3f6; padding: 0 1mm; border-radius: 2px; }
pre { font: 8.5pt/1.4 "Menlo", monospace; background: #f6f8fa; border: 1px solid var(--line); border-radius: 4px;
      padding: 3mm 4mm; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 2mm 0 4mm; font-size: 9pt; break-inside: avoid; width: auto; }
th { background: var(--ink); color: #fff; font-weight: 600; text-align: left; padding: 1.4mm 2.5mm; }
td { border-bottom: 1px solid var(--line); padding: 1.2mm 2.5mm; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fb; }
figure { margin: 3mm 0 5mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 120mm; }
figure.wide img { max-height: 160mm; }
figure.narrow img { max-width: 72%; }
figcaption { font-size: 8.5pt; color: var(--muted); margin-top: 1.5mm; text-align: center; }
figcaption b { color: var(--ink); }
.admonition { border-left: 4px solid var(--accent); background: #eef4f9; padding: 3mm 4mm 1mm; margin: 3mm 0 5mm;
              border-radius: 0 4px 4px 0; break-inside: avoid; }
.admonition.conditions { border-color: #e9a23b; background: #fdf6ea; }
.admonition.note { border-color: #8a96a8; background: #f3f5f8; }
.admonition-title { font-weight: 700; font-size: 10pt; margin-bottom: 1.5mm; color: var(--ink); }
.admonition p:last-child, .admonition ul:last-child { margin-bottom: 2mm; }
.titlepage { height: 245mm; display: flex; flex-direction: column; justify-content: center; break-after: page; }
.titlepage .part { font: 600 11pt "Avenir Next", sans-serif; color: var(--accent); letter-spacing: 0.5pt; text-transform: uppercase; }
.titlepage .title { font-size: 28pt; font-weight: 700; line-height: 1.15; margin: 4mm 0 4mm; }
.titlepage .subtitle { font: 14pt "Avenir Next", sans-serif; color: var(--muted); margin-bottom: 18mm; }
.titlepage .meta { font-size: 10pt; color: var(--muted); line-height: 1.7; }
.titlepage .bar { width: 40mm; height: 3px; background: var(--accent); margin-bottom: 8mm; }
.titlepage .metarow { display: flex; align-items: flex-end; justify-content: space-between; gap: 10mm; }
.titlepage .meta a { color: var(--accent); text-decoration: none; }
.titlepage .qr { width: 34mm; height: auto; flex: none; }
.titlepage + h1, .titlepage + section > h1 { break-before: avoid; }
.toc { font-family: "Avenir Next", sans-serif; font-size: 10pt; }
"""


def render(md_text):
    """Markdown -> (metadata, HTML body) with numbered figures"""
    md = markdown.Markdown(extensions=["meta", "tables", "admonition", "attr_list", "fenced_code", "sane_lists", "toc"])
    body = md.convert(md_text)
    meta = {k: " ".join(v) for k, v in md.Meta.items()}
    return meta, body


def figures(body, counter):
    """<p><img alt=caption src=path></p> -> numbered <figure>; paths relative to the project root.
    Several images in one paragraph -> separate figures, one under another."""
    def one(attrs):
        alt = html.unescape(re.search(r'alt="([^"]*)"', attrs).group(1))
        src = re.search(r'src="([^"]*)"', attrs).group(1)
        cls = re.search(r'class="([^"]*)"', attrs)
        if not src.startswith(("http:", "https:", "file:", "/")):
            path = ROOT / src
            if not path.exists():
                print(f"   missing image: {src}", file=sys.stderr)
            src = path.as_uri()
        counter[0] += 1
        return (f'<figure class="{cls.group(1) if cls else ""}"><img src="{src}" alt="{html.escape(alt)}">'
                f'<figcaption><b>{counter[0]} pav.</b> {html.escape(alt)}</figcaption></figure>')
    return re.sub(r"<p>((?:<img [^>]*?/?>\s*)+)</p>",
                  lambda m: "".join(one(a) for a in re.findall(r"<img ([^>]*?)/?>", m.group(1))), body)


def titlepage(meta):
    sub = f'<div class="subtitle">{html.escape(meta.get("subtitle", ""))}</div>' if meta.get("subtitle") else ""
    return (f'<section class="titlepage"><div class="part">{html.escape(meta.get("part", ""))}</div>'
            f'<div class="title">{html.escape(meta.get("title", ""))}</div>{sub}<div class="bar"></div>'
            f'<div class="metarow"><div class="meta">{html.escape(AUTHOR)}<br>{html.escape(COURSE)}<br>{html.escape(DATE)}<br>'
            f'Kodas ir duomenys: <a href="{REPO}">{html.escape(REPO)}</a></div>'
            + (f'<img class="qr" src="{QR.as_uri()}" alt="QR: {html.escape(REPO)}">' if QR.exists() else "")
            + '</div></section>')


def page(sections, footer):
    return (f'<!doctype html><html lang="lt"><head><meta charset="utf-8"><title>{html.escape(footer)}</title>'
            f'<style>{CSS.replace("var(--footer)", json.dumps(footer, ensure_ascii=False))}</style></head>'
            f'<body>{"".join(sections)}</body></html>')


def build_part(path, counter):
    meta, body = render(path.read_text())
    body = figures(body, counter)
    footer = f'{meta.get("part", "")} · {meta.get("title", "")}'.strip(" ·")
    return meta, f"<section>{titlepage(meta)}{body}</section>", footer


def to_pdf(html_text, tmp, name):
    src = tmp / f"{name}.html"
    src.write_text(html_text)
    pdf = tmp / f"{name}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf}", src.as_uri()], check=True, capture_output=True, timeout=180)
    return pdf


def main():
    parts = sorted(SRC.glob("*.md"))
    if sys.argv[1:]:
        parts = [p for p in parts if p.stem in sys.argv[1:]]
    tmp = Path(tempfile.mkdtemp(prefix="report-"))
    built = []
    for p in parts:
        meta, section, footer = build_part(p, [0])
        built.append(to_pdf(page([section], footer), tmp, p.stem))
        print(f"{p.stem}.pdf")
    if not sys.argv[1:] and len(parts) > 1:
        counter = [0]
        sections = [build_part(p, counter)[1] for p in parts]
        built.append(to_pdf(page(sections, "Lygiagretusis programavimas · ataskaita"), tmp, "ataskaita_visa"))
        print("ataskaita_visa.pdf")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in built:
        shutil.move(str(f), OUT / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved: {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
