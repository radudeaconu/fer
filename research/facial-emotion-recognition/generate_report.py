"""Generate report.md from JSONs in results/ using fields.yaml as schema."""
import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIELDS_FILE = ROOT / "fields.yaml"
OUTLINE_FILE = ROOT / "outline.yaml"
REPORT_FILE = ROOT / "report.md"

# Fields shown in TOC, by item type
TOC_MODEL_FIELDS = ["release_year", "subcategory", "accuracy_rafdb", "param_count"]
TOC_DATASET_FIELDS = ["release_year", "subcategory", "num_images", "in_the_wild_or_lab"]


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify_anchor(name):
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


def is_uncertain_value(v):
    if v is None:
        return True
    if isinstance(v, str):
        if not v.strip():
            return True
        if "[uncertain]" in v.lower():
            return True
    if isinstance(v, list) and not v:
        return True
    if isinstance(v, dict) and not v:
        return True
    return False


def format_value(v, field_name=""):
    """Render a value as compact markdown."""
    if isinstance(v, list):
        if not v:
            return ""
        if all(isinstance(x, dict) for x in v):
            lines = []
            for d in v:
                parts = [f"{k}: {format_value(val)}" for k, val in d.items()
                         if not is_uncertain_value(val)]
                lines.append(" | ".join(parts))
            return "\n  - " + "\n  - ".join(lines)
        joined = ", ".join(str(x) for x in v)
        if len(joined) > 120:
            return "\n  - " + "\n  - ".join(str(x) for x in v)
        return joined
    if isinstance(v, dict):
        parts = [f"**{k}**: {format_value(val)}" for k, val in v.items()
                 if not is_uncertain_value(val)]
        return "; ".join(parts)
    s = str(v).strip()
    return s


def get_field(data, name):
    """Lookup a field, supporting flat or nested-by-category structure."""
    if name in data:
        return data[name]
    for v in data.values():
        if isinstance(v, dict) and name in v:
            return v[name]
    return None


def collect_known_field_names(fields_doc):
    names = set()
    cats = fields_doc.get("categories", {})
    for cat_fields in cats.values():
        for f in cat_fields or []:
            names.add(f["name"])
    return names


def build_toc_line(idx, data, anchor, toc_fields):
    """name (link) - field: value | field: value ..."""
    name = data.get("name", "(unknown)")
    parts = []
    for fname in toc_fields:
        v = get_field(data, fname)
        if is_uncertain_value(v):
            continue
        # for accuracy fields, trim verbose strings
        s = format_value(v, fname)
        if isinstance(s, str) and len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{fname}: {s}")
    extras = " — " + " | ".join(parts) if parts else ""
    return f"{idx}. [{name}](#{anchor}){extras}"


def render_item(data, fields_doc, item_idx):
    """Render the detail section for one item."""
    name = data.get("name", "(unknown)")
    anchor = slugify_anchor(name) + f"-{item_idx}"
    item_type = data.get("type", "unknown")
    uncertain_set = set(data.get("uncertain", []) or [])
    known_names = collect_known_field_names(fields_doc)

    lines = []
    lines.append(f"### <a id=\"{anchor}\"></a>{item_idx}. {name}")
    lines.append("")

    cats = fields_doc.get("categories", {})

    # Order: common, then type-specific, then "other"
    if item_type == "dataset":
        ordered = [("Common", cats.get("common", [])),
                   ("Dataset", cats.get("datasets", []))]
    elif item_type == "model":
        ordered = [("Common", cats.get("common", [])),
                   ("Model", cats.get("models", []))]
    else:
        ordered = [("Common", cats.get("common", []))]

    # Render each category
    for cat_label, cat_fields in ordered:
        rendered_rows = []
        for fdef in cat_fields or []:
            fname = fdef["name"]
            if fname in uncertain_set:
                continue
            v = get_field(data, fname)
            if is_uncertain_value(v):
                continue
            formatted = format_value(v, fname)
            if not formatted:
                continue
            # Long text: blockquote style
            if isinstance(formatted, str) and len(formatted) > 100 and "\n" not in formatted:
                rendered_rows.append(f"- **{fname}**:\n  > {formatted}")
            else:
                rendered_rows.append(f"- **{fname}**: {formatted}")
        if rendered_rows:
            lines.append(f"**{cat_label}**")
            lines.append("")
            lines.extend(rendered_rows)
            lines.append("")

    # Extra fields not in fields.yaml
    skip = known_names | {"uncertain", "_source_file", "type"}
    extras = []
    for k, v in data.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            # nested category bucket — descend and pick fields not in known
            for kk, vv in v.items():
                if kk in skip:
                    continue
                if is_uncertain_value(vv):
                    continue
                extras.append((kk, vv))
            continue
        if is_uncertain_value(v):
            continue
        extras.append((k, v))
    if extras:
        lines.append("**Other Info**")
        lines.append("")
        for k, v in extras:
            lines.append(f"- **{k}**: {format_value(v, k)}")
        lines.append("")

    # Uncertain field list (for transparency)
    if uncertain_set:
        lines.append("**Uncertain (skipped) fields**")
        lines.append("")
        for f in sorted(uncertain_set):
            lines.append(f"- {f}")
        lines.append("")

    return "\n".join(lines), anchor


def main():
    fields_doc = load_yaml(FIELDS_FILE)
    outline = load_yaml(OUTLINE_FILE)
    topic = outline.get("topic", "Research Report")

    # Load all JSONs
    items = []
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"SKIP {fp.name}: {e}", file=sys.stderr)
            continue
        data["_source_file"] = fp.name
        items.append(data)

    # Split into models and datasets, sort by name
    models = sorted(
        [x for x in items if x.get("type") == "model"],
        key=lambda d: d.get("name", "").lower()
    )
    datasets = sorted(
        [x for x in items if x.get("type") == "dataset"],
        key=lambda d: d.get("name", "").lower()
    )
    others = [x for x in items if x.get("type") not in ("model", "dataset")]

    out = []
    out.append(f"# {topic}")
    out.append("")
    out.append(f"_Generated from {len(items)} structured JSON entries in `results/`._")
    out.append("")
    out.append(f"**Total: {len(items)}** ({len(models)} models, {len(datasets)} datasets"
               + (f", {len(others)} other" if others else "") + ")")
    out.append("")
    out.append("---")
    out.append("")

    # ============= TOC =============
    out.append("## Table of Contents")
    out.append("")
    out.append(f"### Models ({len(models)})")
    out.append("")
    detail_blocks = []
    idx = 0
    for d in models:
        idx += 1
        name = d.get("name", "(unknown)")
        anchor = slugify_anchor(name) + f"-{idx}"
        # build toc line
        parts = []
        for fname in TOC_MODEL_FIELDS:
            v = get_field(d, fname)
            if is_uncertain_value(v):
                continue
            s = format_value(v, fname)
            if isinstance(s, str) and len(s) > 60:
                s = s[:57] + "..."
            parts.append(f"{fname}: {s}")
        extras = " — " + " | ".join(parts) if parts else ""
        out.append(f"{idx}. [{name}](#{anchor}){extras}")

    out.append("")
    out.append(f"### Datasets ({len(datasets)})")
    out.append("")
    for d in datasets:
        idx += 1
        name = d.get("name", "(unknown)")
        anchor = slugify_anchor(name) + f"-{idx}"
        parts = []
        for fname in TOC_DATASET_FIELDS:
            v = get_field(d, fname)
            if is_uncertain_value(v):
                continue
            s = format_value(v, fname)
            if isinstance(s, str) and len(s) > 60:
                s = s[:57] + "..."
            parts.append(f"{fname}: {s}")
        extras = " — " + " | ".join(parts) if parts else ""
        out.append(f"{idx}. [{name}](#{anchor}){extras}")

    if others:
        out.append("")
        out.append(f"### Other ({len(others)})")
        out.append("")
        for d in others:
            idx += 1
            name = d.get("name", "(unknown)")
            anchor = slugify_anchor(name) + f"-{idx}"
            out.append(f"{idx}. [{name}](#{anchor})")

    out.append("")
    out.append("---")
    out.append("")

    # ============= DETAILS =============
    out.append("## Models")
    out.append("")
    item_idx = 0
    for d in models:
        item_idx += 1
        block, _ = render_item(d, fields_doc, item_idx)
        out.append(block)
        out.append("---")
        out.append("")

    out.append("## Datasets")
    out.append("")
    for d in datasets:
        item_idx += 1
        block, _ = render_item(d, fields_doc, item_idx)
        out.append(block)
        out.append("---")
        out.append("")

    if others:
        out.append("## Other")
        out.append("")
        for d in others:
            item_idx += 1
            block, _ = render_item(d, fields_doc, item_idx)
            out.append(block)
            out.append("---")
            out.append("")

    REPORT_FILE.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote report: {REPORT_FILE}")
    print(f"Items: {len(items)} ({len(models)} models, {len(datasets)} datasets)")


if __name__ == "__main__":
    main()
