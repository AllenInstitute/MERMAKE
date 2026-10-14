"""mkdocs hook: render the configuration reference from the settings dataclasses.

Replaces the marker ``<!-- config-reference -->`` in a page with tables generated from
``mermake.config``, so the documentation cannot drift from the code. ``mermake.config``
does not import cupy, so this works without a GPU.
"""
from mermake.config import reference_rows

MARKER = "<!-- config-reference -->"


def _cell(text):
	return str(text).replace("|", "\\|").replace("\n", " ")


def _value(value):
	return "" if value is None else f"`{value}`"


def render():
	rows = reference_rows()
	out = ["### `[paths]`", "",
	       "| Key | Type | Default | Description |", "|---|---|---|---|"]
	for r in (r for r in rows if r["section"] == "paths"):
		default = "**required**" if r["required"] else (_value(r["default"]) or "none")
		out.append(f"| `{r['key']}` | {r['type']} | {default} | {_cell(r['doc'])} |")
	out += ["", "### `[hybs]` and `[dapi]`", "",
	        "Both sections take the same keys. They only differ in their defaults.", "",
	        "| Key | Type | Unit | Default `[hybs]` | Default `[dapi]` | Description |",
	        "|---|---|---|---|---|---|"]
	dapi = {r["key"]: r for r in rows if r["section"] == "dapi"}
	for r in (r for r in rows if r["section"] == "hybs"):
		out.append(f"| `{r['key']}` | {r['type']} | {r['unit'] or ''} | {_value(r['default'])} | "
		           f"{_value(dapi[r['key']]['default'])} | {_cell(r['doc'])} |")
	out += ["", "### `[output]`", "",
	        "| Key | Type | Default | Description |", "|---|---|---|---|"]
	for r in (r for r in rows if r["section"] == "output"):
		out.append(f"| `{r['key']}` | {r['type']} | {_value(r['default'])} | {_cell(r['doc'])} |")
	return "\n".join(out)


def on_page_markdown(markdown, **kwargs):
	return markdown.replace(MARKER, render())
