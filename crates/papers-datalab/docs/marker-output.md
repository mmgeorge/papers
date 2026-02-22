# Datalab Marker JSON Output Reference

A guide to the block types emitted by [Datalab's Marker API](https://documentation.datalab.to/api-reference/marker) in JSON output format, with abbreviated examples from a real parsed academic paper and strategies for handling images/figures/captions in RAG pipelines.

---

## Document Structure Overview

Marker's JSON output is a tree of blocks. The root node contains `Page` blocks, each of which contains the content blocks for that page. There are no explicit chapter/section containers — instead, each block carries a `section_hierarchy` dict that maps heading depth to the relevant `SectionHeader` block ID.

```
Document (root)
└── Page (×N)
    ├── PageHeader
    ├── SectionHeader        ← h1–h6 headings
    ├── Text                 ← body paragraphs
    ├── Equation             ← display math (LaTeX)
    ├── Picture / Figure     ← extracted images + AI-generated alt text
    ├── Caption              ← author-written figure/table captions
    ├── ListGroup            ← bulleted / numbered lists
    ├── Table                ← HTML tables with full row/col structure
    └── PageFooter
```

### Common Fields (all block types)

| Field | Type | Description |
|-------|------|-------------|
| `block_type` | string | The type identifier (e.g. `"Text"`, `"Picture"`) |
| `id` | string | Unique path-style ID: `/page/{N}/{Type}/{M}` |
| `bbox` | `[x1, y1, x2, y2]` | Bounding box in page coordinates |
| `polygon` | `[[x,y], ...]` | 4-corner polygon (clockwise from top-left) |
| `page` | int | Zero-indexed page number |
| `html` | string | HTML representation of the block content |
| `section_hierarchy` | dict | Maps heading depth `"1"`,`"2"`,`"3"` → SectionHeader IDs |
| `images` | dict | Map of `filename → base64` data (only on Picture/Figure) |
| `markdown` | string or null | Markdown representation (often null in JSON output) |
| `children` | array | Child blocks (only on Page and root) |

---

## Block Types with Examples

### Page

Container for all blocks on a single page. The root document node contains an array of these.

```json
{
  "block_type": "Page",
  "id": "/page/0/Page/0",
  "bbox": [0.0, 0.0, 1632.0, 2112.0],
  "polygon": [[0.0, 0.0], [1632.0, 0.0], [1632.0, 2112.0], [0.0, 2112.0]],
  "children": ["...23 child blocks..."]
}
```

**Notes:** `html` on a Page block is the concatenated HTML of all its children. The `page` field is null on Page blocks — child blocks carry their own `page` index.

---

### SectionHeader

Document headings at various levels (`<h1>` through `<h6>`). Use the HTML tag to determine depth.

```json
{
  "block_type": "SectionHeader",
  "id": "/page/0/SectionHeader/1",
  "bbox": [132.0, 209.0, 538.0, 253.0],
  "page": 0,
  "html": "<h1>Vertex Block Descent</h1>"
}
```

```json
{
  "block_type": "SectionHeader",
  "id": "/page/2/SectionHeader/8",
  "bbox": [838.0, 490.0, 1181.0, 522.0],
  "page": 2,
  "html": "<h3>\n3.1 Global Optimization\n</h3>",
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/2/SectionHeader/4"
  }
}
```

**Notes:** These are the anchors referenced by `section_hierarchy` on other blocks. The `section_hierarchy` keys (`"1"`, `"2"`, `"3"`) correspond to `<h1>`, `<h2>`, `<h3>` depth levels. There are no nested chapter containers — reconstruct hierarchy by grouping blocks that share the same `section_hierarchy` values.

---

### Text

Body paragraphs — the most common block type.

```json
{
  "block_type": "Text",
  "id": "/page/0/Text/2",
  "bbox": [132.0, 272.0, 641.0, 306.0],
  "page": 0,
  "html": "<p>ANKA HE CHEN, University of Utah, USA</p>",
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1"
  }
}
```

**Notes:** May contain inline math as `<math>` tags within the `<p>` element.

---

### Picture

Extracted raster images with AI-generated descriptions and base64 image data.

```json
{
  "block_type": "Picture",
  "id": "/page/0/Picture/0",
  "bbox": [0.0, 181.0, 68.0, 268.0],
  "page": 0,
  "html": "<img alt=\"Check for updates icon\" src=\"2dfa6ac3...img.jpg\"/><div class=\"img-description\" style=\"border: 1px solid #ccc; padding: 10px;\"><div class=\"img-alt\">Check for updates icon</div></div>",
  "images": {
    "2dfa6ac3edfe874f68aa0cbccaa42322_img.jpg": "/9j/4AAQSkZJRgABAQAAAQABAAD/..."
  }
}
```

**HTML structure breakdown:**

- `<img alt="...">` — the `alt` attribute contains the **AI-generated caption** describing the visual content
- `<div class="img-description">` — wrapper div for the AI caption display
- `<div class="img-alt">` — the AI caption text (same content as `alt`)
- `images` dict — keys are filenames, values are base64-encoded JPEG/PNG data

**Notes:** The AI-generated captions describe what the image *visually looks like* (colors, shapes, layout). These are distinct from the author-written `Caption` blocks. Disable with `disable_image_captions=true` in the API call.

---

### Figure

Complex visuals (charts, diagrams, graphs). Same structure as `Picture`.

```json
{
  "block_type": "Figure",
  "id": "/page/5/Figure/10",
  "bbox": [838.0, 211.0, 1506.0, 426.0],
  "page": 5,
  "html": "<img alt=\"Figure 6: Visualization of the gravity initialization factor (a-tilde) for a swinging elastic pendulum. The color scale ranges from 0.0 (blue) to 1.0 (red).\" src=\"29ac39bfd...img.jpg\"/><div class=\"img-description\" style=\"border: 1px solid #ccc; padding: 10px;\"><div class=\"img-alt\">Figure 6: Visualization of the gravity...</div></div>",
  "images": {
    "29ac39bfd74e57a92045649f83cad949_img.jpg": "/9j/4AAQSkZJRgABAQAAAQABAAD/..."
  },
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/2/SectionHeader/4",
    "3": "/page/4/SectionHeader/27"
  }
}
```

**Notes:** The distinction between `Picture` and `Figure` appears to be that `Figure` is used for more structured/complex visuals, while `Picture` is used for raster images and photos. Both receive AI captions and base64 image data identically.

---

### Caption

Author-written figure and table captions extracted from the PDF text. These are separate blocks — *not* embedded in the Picture/Figure block.

```json
{
  "block_type": "Caption",
  "id": "/page/0/Caption/7",
  "bbox": [143.0, 880.0, 1506.0, 912.0],
  "page": 0,
  "html": "<p><b>Fig. 1.</b> Example simulation results using our solver, both of those methods involve more than 100 million DoFs and 1 million active collisions.</p>",
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1"
  }
}
```

**Notes:** Caption blocks are typically the immediate sibling *after* their associated Picture/Figure block in the page's children array. They do not have an `images` dict. The content is the original text from the document authors.

---

### Equation

Display math rendered as LaTeX within `<math>` tags.

```json
{
  "block_type": "Equation",
  "id": "/page/2/Equation/10",
  "bbox": [1051.0, 612.0, 1506.0, 661.0],
  "page": 2,
  "html": "<p>\n<math display=\"block\">\\mathbf{x}^{t+1} = \\underset{\\mathbf{x}}{\\operatorname{argmin}} G(\\mathbf{x}), \\quad (1)</math>\n</p>",
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/2/SectionHeader/4",
    "3": "/page/2/SectionHeader/8"
  }
}
```

**Notes:** Inline math may also appear within `Text` blocks as `<math>` tags. `Equation` blocks are specifically for display/block-level math.

---

### ListGroup

Bulleted or numbered lists.

```json
{
  "block_type": "ListGroup",
  "id": "/page/4/ListGroup/12",
  "bbox": [169.0, 1520.0, 789.0, 1784.0],
  "page": 4,
  "html": "\n<ul style=\"list-style-type: none\">\n<li>• Edge-edge collisions use continuous collision detection (CCD). <math>\\mathbf{x}_a</math> and <math>\\mathbf{x}_b</math> correspond to the intersection points on either edge...</li>\n<li>• Vertex-face collisions...</li>\n</ul>",
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/2/SectionHeader/4",
    "3": "/page/4/SectionHeader/7"
  }
}
```

**Notes:** List items may contain inline math. The `list-style-type: none` with bullet characters in the text is a Marker convention.

---

### Table

Full HTML tables with row and column structure.

```json
{
  "block_type": "Table",
  "id": "/page/11/Table/4",
  "bbox": [132.0, 1161.0, 1504.0, 1503.0],
  "page": 11,
  "html": "\n<table border=\"1\">\n<thead>\n<tr>\n<th rowspan=\"2\">Experiment Name</th>\n<th rowspan=\"2\">Number of<br/>Vert.</th>\n<th rowspan=\"2\">Tet.</th>\n<th rowspan=\"2\">Color</th>\n<th rowspan=\"2\">Material<br/>Type</th>\n<th rowspan=\"2\">Stiffness</th>\n...</tr>\n</thead>\n<tbody>...</tbody>\n</table>",
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/7/SectionHeader/13",
    "3": "/page/10/SectionHeader/6"
  }
}
```

**Notes:** Tables include `<thead>`, `<tbody>`, `rowspan`, `colspan` attributes as needed. Like figures, tables often have an adjacent `Caption` block as a sibling.

---

### PageHeader

Running header content (journal name, paper title, etc.).

```json
{
  "block_type": "PageHeader",
  "id": "/page/1/PageHeader/0",
  "bbox": [133.0, 149.0, 634.0, 177.0],
  "page": 1,
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/0/SectionHeader/17"
  }
}
```

**Notes:** Often has empty or minimal `html`. Generally noise for RAG purposes.

---

### PageFooter

Footer content (page numbers, journal citation info).

```json
{
  "block_type": "PageFooter",
  "id": "/page/0/PageFooter/22",
  "bbox": [915.0, 1883.0, 1522.0, 1911.0],
  "page": 0,
  "section_hierarchy": {
    "1": "/page/0/SectionHeader/1",
    "2": "/page/0/SectionHeader/17"
  }
}
```

**Notes:** Generally noise for RAG purposes. Can be filtered out during ingestion.

---

## Block Type Summary

| Block Type | Count* | Contains Images | Contains Text | RAG Relevance |
|------------|--------|-----------------|---------------|---------------|
| Text | 145 | No | Yes | High — primary content |
| SectionHeader | 30 | No | Yes | High — structure & chunking |
| Caption | 25 | No | Yes | High — figure/table descriptions |
| Picture | 22 | Yes (base64 + AI alt) | AI-generated only | Medium — visual context |
| Equation | 20 | No | LaTeX | Medium — domain-specific |
| Page | 16 | No | Aggregated | Low — container only |
| PageFooter | 16 | No | Minimal | Low — filter out |
| PageHeader | 15 | No | Minimal | Low — filter out |
| ListGroup | 4 | No | Yes | High — structured content |
| Figure | 4 | Yes (base64 + AI alt) | AI-generated only | Medium — visual context |
| Table | 1 | No | Yes (HTML) | High — structured data |

*Counts from example document (16-page academic paper).*

---

## Handling Images, Figures, and Captions for RAG

### Understanding the Two Caption Sources

Every image in Marker's output can have **two distinct textual descriptions**:

1. **AI-generated caption** — Inside the `Picture`/`Figure` block's `html` field as the `alt` attribute and within the `<div class="img-description">` wrapper. Describes what the image *visually looks like*.

   > *"Figure 1: Example simulation results. Left: A glass teapot filled with a dense, magenta-colored, spiky fluid simulation..."*

2. **Author-written caption** — A separate `Caption` block that is a sibling of the `Picture`/`Figure` block. Contains the original caption text as written by the document authors.

   > *"**Fig. 1.** Example simulation results using our solver, both of those methods involve more than 100 million DoFs..."*

### Strategy 1: Associate Captions with their Picture/Figure

The JSON places `Caption` blocks as siblings immediately following their associated `Picture`/`Figure` block. Pair them by adjacency:

```python
def pair_figures_with_captions(page_children):
    """Walk page children and pair Picture/Figure blocks with their Caption."""
    pairs = []
    for i, block in enumerate(page_children):
        if block["block_type"] in ("Picture", "Figure"):
            caption_block = None
            if i + 1 < len(page_children):
                next_block = page_children[i + 1]
                if next_block["block_type"] == "Caption":
                    caption_block = next_block
            pairs.append({
                "figure": block,
                "caption": caption_block,
            })
    return pairs
```

### Strategy 2: Build Combined Figure Chunks for Embedding

Combine all available text signals into a single chunk per figure for maximum retrieval coverage:

```python
import re

def extract_alt_text(html):
    """Extract AI-generated alt text from Picture/Figure HTML."""
    match = re.search(r'alt="([^"]+)"', html or "")
    return match.group(1) if match else ""

def extract_caption_text(caption_block):
    """Extract plain text from a Caption block's HTML."""
    if not caption_block:
        return ""
    html = caption_block.get("html", "") or ""
    # Strip HTML tags for plain text
    return re.sub(r"<[^>]+>", "", html).strip()

def resolve_section_path(block, all_headers):
    """Resolve section_hierarchy to readable section path."""
    sh = block.get("section_hierarchy", {})
    parts = []
    for depth in sorted(sh.keys()):
        header_id = sh[depth]
        if header_id in all_headers:
            parts.append(all_headers[header_id])
    return " > ".join(parts)

def build_figure_chunk(figure_block, caption_block, section_path):
    """Create a combined text chunk for embedding."""
    ai_alt = extract_alt_text(figure_block.get("html", ""))
    author_caption = extract_caption_text(caption_block)

    chunk_text = f"[Figure in section: {section_path}]\n"
    if author_caption:
        chunk_text += f"Caption: {author_caption}\n"
    if ai_alt:
        chunk_text += f"Visual description: {ai_alt}\n"

    return chunk_text.strip()
```

This gives your embeddings both the technical meaning (author caption) and the visual content (AI description) in a single retrievable unit.

### Strategy 3: Store Images Separately for Display

Save base64 image data to disk or object storage, keyed by block ID. When a figure chunk is retrieved, serve the image alongside the text:

```python
import base64
import os

def store_images(figure_block, output_dir):
    """Save base64 images to disk, return file paths."""
    paths = {}
    images = figure_block.get("images", {})
    for filename, b64_data in images.items():
        block_id = figure_block["id"].replace("/", "_")
        out_path = os.path.join(output_dir, f"{block_id}_{filename}")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        paths[filename] = out_path
    return paths
```

### Strategy 4: Carry Rich Metadata on Every Chunk

Attach metadata for filtering, citation tracking, and UI rendering:

```python
def build_chunk_metadata(block, chunk_type, section_path, image_paths=None):
    """Build metadata dict for a RAG chunk."""
    return {
        "block_type": chunk_type,         # "figure", "text", "table", "equation"
        "block_id": block["id"],          # e.g. "/page/5/Figure/10"
        "page": block.get("page"),        # zero-indexed page number
        "bbox": block.get("bbox"),        # for PDF highlight / citation overlay
        "section": section_path,          # e.g. "Vertex Block Descent > 3 VBD... > 3.7 Init..."
        "has_image": bool(image_paths),   # for filtering queries
        "image_paths": image_paths or [], # for serving images at retrieval time
    }
```

### Strategy 5: Complete Ingestion Pipeline

Putting it all together:

```python
def ingest_marker_json(data, output_dir="./images"):
    """Full ingestion pipeline for Marker JSON output."""
    os.makedirs(output_dir, exist_ok=True)
    chunks = []

    # First pass: collect all section headers for path resolution
    all_headers = {}
    def collect_headers(obj):
        if isinstance(obj, dict):
            if obj.get("block_type") == "SectionHeader":
                text = re.sub(r"<[^>]+>", "", obj.get("html", "")).strip()
                all_headers[obj["id"]] = text
            for child in obj.get("children", []):
                collect_headers(child)
    collect_headers(data)

    # Second pass: process each page
    for page_block in data.get("children", []):
        children = page_block.get("children", [])
        skip_next = False

        for i, block in enumerate(children):
            if skip_next:
                skip_next = False
                continue

            bt = block.get("block_type", "")
            section_path = resolve_section_path(block, all_headers)

            # Skip noise blocks
            if bt in ("PageHeader", "PageFooter", "Page"):
                continue

            # Handle figures: combine with adjacent caption
            if bt in ("Picture", "Figure"):
                caption_block = None
                if i + 1 < len(children):
                    nxt = children[i + 1]
                    if nxt.get("block_type") == "Caption":
                        caption_block = nxt
                        skip_next = True  # don't double-process the caption

                image_paths = store_images(block, output_dir)
                chunk_text = build_figure_chunk(block, caption_block, section_path)
                metadata = build_chunk_metadata(
                    block, "figure", section_path, list(image_paths.values())
                )
                chunks.append({"text": chunk_text, "metadata": metadata})
                continue

            # Handle text, equations, lists, tables
            if bt in ("Text", "Equation", "ListGroup", "Table", "SectionHeader"):
                text = re.sub(r"<[^>]+>", "", block.get("html", "") or "").strip()
                if not text:
                    continue
                metadata = build_chunk_metadata(block, bt.lower(), section_path)
                chunks.append({"text": text, "metadata": metadata})

    return chunks
```

### Decision Matrix: What to Do with Image Data

| Approach | When to Use | Trade-off |
|----------|-------------|-----------|
| **Store images, embed text only** | Most RAG pipelines. Users search by text, images displayed alongside results. | Cheapest. No multimodal model needed. |
| **Embed images with CLIP/multimodal model** | Users might search by visual similarity ("show me the graph with colored spheres"). | More compute and storage. Requires multimodal embedding model. |
| **Drop base64 entirely** | Text-only retrieval, no need to display images. Storage-constrained. | Lose ability to show images in results. |
| **Disable AI captions at API level** | You only need author captions, or you generate your own descriptions. Set `disable_image_captions=true`. | Lose visual description signal for embedding. |

### Recommended Defaults

- **Always pair** Picture/Figure with adjacent Caption blocks.
- **Always embed** the combined author caption + AI alt text for maximum recall.
- **Always store** images to disk/S3 keyed by block ID for display at retrieval time.
- **Always drop** PageHeader and PageFooter blocks — they are noise.
- **Always carry** `block_id`, `page`, and `bbox` in metadata for citation tracking back to the source PDF.
