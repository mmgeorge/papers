"""Run TableFormer ONNX inference and verify against expected output."""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common.preprocess import preprocess

# OTSL vocabulary (from tm_config.json)
PAD, UNK, START, END = 0, 1, 2, 3
ECEL, FCEL, LCEL, UCEL, XCEL = 4, 5, 6, 7, 8
NL, CHED, RHED, SROW = 9, 10, 11, 12

TAG_NAMES = {
    0: "<pad>", 1: "<unk>", 2: "<start>", 3: "<end>",
    4: "ecel", 5: "fcel", 6: "lcel", 7: "ucel", 8: "xcel",
    9: "nl", 10: "ched", 11: "rhed", 12: "srow",
}

CELL_TOKENS = {FCEL, ECEL, CHED, RHED, SROW}
MAX_SEQ = 512


def decode(enc_sess, dec_sess, pixel_values):
    """Run encoder + autoregressive decoder loop."""
    # Encoder
    enc_mem, enc_raw = enc_sess.run(None, {"pixel_values": pixel_values})
    print(f"  encoder_memory: {enc_mem.shape}, enc_out_raw: {enc_raw.shape}")

    # Get decoder metadata from session
    dec_inputs = {inp.name: inp for inp in dec_sess.get_inputs()}
    kv_shape = list(dec_inputs["past_key_values.0.key"].shape)
    n_layers = sum(1 for name in dec_inputs if name.endswith(".key"))

    # Init KV cache
    past_kv = {}
    for layer in range(n_layers):
        past_kv[f"past_key_values.{layer}.key"] = np.zeros(kv_shape, dtype=np.float32)
        past_kv[f"past_key_values.{layer}.value"] = np.zeros(kv_shape, dtype=np.float32)

    # Decode loop
    input_ids = np.array([[START]], dtype=np.int64)
    tokens = [START]
    cell_hidden_states = []

    skip_next_tag = True
    prev_tag_ucel = False
    line_num = 0
    first_lcel = True
    bboxes_to_merge = {}
    cur_bbox_ind = -1
    bbox_ind = 0

    for step in range(MAX_SEQ):
        feed = {
            "input_ids": input_ids,
            "encoder_memory": enc_mem,
            "step": np.array([step], dtype=np.int64),
            **past_kv,
        }

        outputs = dec_sess.run(None, feed)
        logits = outputs[0]       # [1, 1, 13]
        hidden = outputs[1]       # [1, 1, 512]

        new_tag = int(np.argmax(logits[0, 0, :]))

        # Structure correction
        if line_num == 0 and new_tag == XCEL:
            new_tag = LCEL
        if prev_tag_ucel and new_tag == LCEL:
            new_tag = FCEL

        # Collect hidden states for bbox prediction (matching docling logic)
        if not skip_next_tag:
            if new_tag in (FCEL, ECEL, CHED, RHED, SROW, NL, UCEL):
                cell_hidden_states.append(hidden[0, 0, :].copy())
                if not first_lcel:
                    bboxes_to_merge[cur_bbox_ind] = bbox_ind
                bbox_ind += 1

        if new_tag != LCEL:
            first_lcel = True
        else:
            if first_lcel:
                cell_hidden_states.append(hidden[0, 0, :].copy())
                first_lcel = False
                cur_bbox_ind = bbox_ind
                bboxes_to_merge[cur_bbox_ind] = -1
                bbox_ind += 1

        if new_tag in (NL, UCEL, XCEL):
            skip_next_tag = True
        else:
            skip_next_tag = False

        if new_tag == UCEL:
            prev_tag_ucel = True
        else:
            prev_tag_ucel = False

        if new_tag == NL:
            line_num += 1

        tokens.append(new_tag)

        if new_tag == END:
            break

        # Swap present KV → past KV
        for layer in range(n_layers):
            past_kv[f"past_key_values.{layer}.key"] = outputs[2 + layer * 2]
            past_kv[f"past_key_values.{layer}.value"] = outputs[2 + layer * 2 + 1]

        input_ids = np.array([[new_tag]], dtype=np.int64)

    return tokens, cell_hidden_states, bboxes_to_merge, enc_raw


def predict_bboxes(bbox_sess, enc_raw, cell_hidden_states, bboxes_to_merge):
    """Run bbox decoder and merge span bboxes."""
    if len(cell_hidden_states) == 0:
        return np.empty((0, 4)), np.empty((0, 3))

    cell_hidden = np.stack(cell_hidden_states, axis=0)  # [N, 512]
    bboxes, classes = bbox_sess.run(
        None, {"enc_out_raw": enc_raw, "cell_hidden_states": cell_hidden}
    )

    # Merge span bboxes (cxcywh format)
    merged_bboxes = []
    merged_classes = []
    boxes_to_skip = set()

    for i in range(len(bboxes)):
        if i in bboxes_to_merge:
            end_idx = bboxes_to_merge[i]
            if end_idx >= 0 and end_idx < len(bboxes):
                box1 = bboxes[i]
                box2 = bboxes[end_idx]
                # Merge: union of cxcywh boxes
                new_w = (box2[0] + box2[2] / 2) - (box1[0] - box1[2] / 2)
                new_h = (box2[1] + box2[3] / 2) - (box1[1] - box1[3] / 2)
                new_left = box1[0] - box1[2] / 2
                new_top = min(box2[1] - box2[3] / 2, box1[1] - box1[3] / 2)
                merged_bboxes.append([
                    new_left + new_w / 2, new_top + new_h / 2, new_w, new_h
                ])
                merged_classes.append(classes[i])
                boxes_to_skip.add(end_idx)
            else:
                merged_bboxes.append(bboxes[i].tolist())
                merged_classes.append(classes[i])
        elif i not in boxes_to_skip:
            merged_bboxes.append(bboxes[i].tolist())
            merged_classes.append(classes[i])

    if merged_bboxes:
        return np.array(merged_bboxes), np.array(merged_classes)
    return np.empty((0, 4)), np.empty((0, 3))


def cxcywh_to_xyxy(bboxes):
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    result = np.zeros_like(bboxes)
    result[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x1
    result[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y1
    result[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x2
    result[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y2
    return result


def otsl_to_html(tag_ids):
    """Convert OTSL token IDs to HTML table structure."""
    from itertools import groupby

    # Convert IDs to tag names, strip control tokens
    tags = []
    for t in tag_ids:
        name = TAG_NAMES.get(t, "")
        if name not in ("<pad>", "<unk>", "<start>", "<end>"):
            tags.append(name)

    if not tags:
        return ""

    # Split on "nl" → rows
    rows = [list(g) for k, g in groupby(tags, lambda x: x == "nl") if not k]
    if not rows:
        return ""

    # Pad to square
    max_cols = max(len(r) for r in rows)
    for r in rows:
        r.extend(["lcel"] * (max_cols - len(r)))

    # Build HTML
    html_parts = ["<table>"]
    thead_open = False

    for row_idx, row in enumerate(rows):
        if not thead_open and "ched" in row:
            html_parts.append("<thead>")
            thead_open = True
        if thead_open and "ched" not in row:
            html_parts.append("</thead>")
            thead_open = False

        html_parts.append("<tr>")
        for col_idx, cell in enumerate(row):
            if cell in ("fcel", "ched", "rhed", "srow", "ecel"):
                colspan = 1
                rowspan = 1

                # Check right for lcel
                c = col_idx + 1
                while c < len(row) and row[c] == "lcel":
                    colspan += 1
                    c += 1

                # Check right for xcel (2D span)
                if col_idx + 1 < len(row) and row[col_idx + 1] == "xcel":
                    colspan = 1
                    c = col_idx + 1
                    while c < len(row) and row[c] == "xcel":
                        colspan += 1
                        c += 1
                    # Check down for ucel
                    rowspan = 1
                    r = row_idx + 1
                    while r < len(rows) and rows[r][col_idx] == "ucel":
                        rowspan += 1
                        r += 1
                else:
                    # Check down for ucel
                    r = row_idx + 1
                    while r < len(rows) and rows[r][col_idx] == "ucel":
                        rowspan += 1
                        r += 1

                tag = "th" if cell == "ched" else "td"
                attrs = ""
                if colspan > 1:
                    attrs += f' colspan="{colspan}"'
                if rowspan > 1:
                    attrs += f' rowspan="{rowspan}"'
                html_parts.append(f"<{tag}{attrs}></{tag}>")

        html_parts.append("</tr>")

    if thead_open:
        html_parts.append("</thead>")
    html_parts.append("</table>")
    return "".join(html_parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_onnx.py <image_path> [model_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")

    # Load ONNX sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("Loading ONNX models...")
    enc_sess = ort.InferenceSession(
        str(model_dir / "encoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        str(model_dir / "decoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    bbox_sess = ort.InferenceSession(
        str(model_dir / "bbox_decoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )

    # Preprocess
    print(f"Processing: {image_path}")
    pixel_values = preprocess(image_path)
    print(f"  input shape: {pixel_values.shape}")

    # Decode
    print("Decoding...")
    tokens, cell_hidden_states, bboxes_to_merge, enc_raw = decode(
        enc_sess, dec_sess, pixel_values
    )

    # Print OTSL sequence
    tag_names = [TAG_NAMES.get(t, f"?{t}") for t in tokens]
    print(f"\nOTSL sequence ({len(tokens)} tokens):")
    print(" ".join(tag_names))

    # Count structure
    n_cells = sum(1 for t in tokens if t in CELL_TOKENS)
    n_rows = sum(1 for t in tokens if t == NL)
    print(f"\nCells: {n_cells}, Rows: {n_rows}")
    print(f"Hidden states collected: {len(cell_hidden_states)}")
    print(f"Spans to merge: {bboxes_to_merge}")

    # BBox prediction
    print("\nPredicting bboxes...")
    bboxes, classes = predict_bboxes(bbox_sess, enc_raw, cell_hidden_states, bboxes_to_merge)

    if len(bboxes) > 0:
        bboxes_xyxy = cxcywh_to_xyxy(bboxes)
        class_ids = np.argmax(classes, axis=1)
        print(f"Bboxes: {len(bboxes_xyxy)} (after merge)")
        for i, (bbox, cls) in enumerate(zip(bboxes_xyxy, class_ids)):
            print(f"  [{i}] class={cls} bbox=[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")

    # HTML structure
    html = otsl_to_html(tokens)
    print(f"\nHTML structure:")
    print(html)


if __name__ == "__main__":
    main()
