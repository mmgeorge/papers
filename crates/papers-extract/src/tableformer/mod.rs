//! TableFormer V1 table structure recognition — OTSL token sequence → HTML skeleton.
//!
//! Split ONNX model:
//!   - `encoder.onnx` — ResNet-18 + TransformerEncoder (runs once per table)
//!   - `decoder.onnx` — autoregressive decoder with fixed-size KV cache (~200 steps)
//!   - `bbox_decoder.onnx` — attention-gated MLP for cell bounding boxes (runs once)
//!
//! ## CUDA backend (Windows, NVIDIA GPU)
//! Persistent IoBinding, pre-allocated GPU buffers, ORT built-in CUDA graph.
//!
//! ## CPU/CoreML backend (all platforms)
//! session.run()-based decode loop, ndarray KV swap per step.

#[cfg(target_os = "windows")]
mod cuda;
pub mod decode;
mod other;
pub mod otsl;
pub mod preprocess;

use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;
use crate::pdf::PdfChar;
use crate::text;

use decode::{extract_decoder_params, DecoderParams};

/// Result of a single table prediction — HTML skeleton + per-cell bboxes.
pub struct TablePrediction {
    /// HTML table skeleton with empty cells (or filled if `fill_table_html` was called).
    pub html: String,
    /// Per-cell bboxes in xyxy normalized [0,1] within the table crop.
    /// One per `<td>`/`<th>` in HTML emission order. `None` if skipped.
    pub cell_bboxes: Vec<Option<[f32; 4]>>,
}

// ── Backend enum ─────────────────────────────────────────────────────

enum ActiveBackend {
    #[cfg(target_os = "windows")]
    Cuda {
        decoder: Mutex<cuda::DecoderState>,
        #[allow(dead_code)]
        cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    },
    Other {
        decoder: Mutex<Session>,
    },
}

/// TableFormer predictor — holds encoder, decoder, and bbox decoder sessions.
pub struct TableFormerPredictor {
    encoder: Mutex<Session>,
    bbox_decoder: Mutex<Session>,
    decoder_params: DecoderParams,
    backend: ActiveBackend,
}

impl TableFormerPredictor {
    /// Load the three ONNX models and extract decoder parameters.
    pub fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        bbox_decoder_path: &Path,
    ) -> Result<Self, ExtractError> {
        // Encoder + bbox decoder: use platform-appropriate EP
        let encoder_ep = Self::make_ep(false);
        let encoder = build_session(encoder_path, GraphOptimizationLevel::Level3, encoder_ep)?;

        let bbox_ep = Self::make_ep(false);
        let bbox_decoder =
            build_session(bbox_decoder_path, GraphOptimizationLevel::Level3, bbox_ep)?;

        // Build backend-specific decoder state
        let (backend, decoder_params) = Self::build_backend(decoder_path)?;

        let backend_name = match &backend {
            #[cfg(target_os = "windows")]
            ActiveBackend::Cuda { .. } => "CUDA",
            ActiveBackend::Other { .. } => {
                if cfg!(target_os = "macos") {
                    "CoreML"
                } else {
                    "CPU"
                }
            }
        };

        tracing::info!(
            "TableFormer decoder: {} layers, {} heads, max_seq={}, head_dim={}, backend={}",
            decoder_params.num_layers,
            decoder_params.num_heads,
            decoder_params.max_seq,
            decoder_params.head_dim,
            backend_name,
        );
        Ok(Self {
            encoder: Mutex::new(encoder),
            bbox_decoder: Mutex::new(bbox_decoder),
            decoder_params,
            backend,
        })
    }

    fn build_backend(
        decoder_path: &Path,
    ) -> Result<(ActiveBackend, DecoderParams), ExtractError> {
        #[cfg(target_os = "windows")]
        {
            // Try CUDA backend
            let cuda_ctx = cudarc::driver::CudaContext::new(0);
            if let Ok(cuda_ctx) = cuda_ctx {
                tracing::info!("TableFormer: using CUDA backend");

                let cuda_mem = ort::memory::MemoryInfo::new(
                    ort::memory::AllocationDevice::CUDA,
                    0,
                    ort::memory::AllocatorType::Device,
                    ort::memory::MemoryType::Default,
                )
                .map_err(|e| ExtractError::Model(format!("CUDA MemoryInfo: {e}")))?;

                let decoder_ep = ort::execution_providers::CUDAExecutionProvider::default()
                    .with_cuda_graph(true)
                    .build();
                let decoder_session = build_session(
                    decoder_path,
                    GraphOptimizationLevel::Level3,
                    decoder_ep,
                )?;

                let decoder_params = extract_decoder_params(&decoder_session)?;

                // encoder_memory shape: [1, 784, 512]
                // 784 = 28*28 spatial positions, 512 = d_model
                let enc_memory_shape = [1usize, 784, 512];

                let decoder_state = cuda::build_decoder_state(
                    decoder_session,
                    &cuda_mem,
                    &decoder_params,
                    &enc_memory_shape,
                )?;

                return Ok((
                    ActiveBackend::Cuda {
                        decoder: Mutex::new(decoder_state),
                        cuda_ctx,
                    },
                    decoder_params,
                ));
            }
            tracing::info!("TableFormer: CUDA unavailable, falling back to CPU");
        }

        // CPU/CoreML backend
        let decoder_ep = Self::make_ep(false);
        let decoder_session =
            build_session(decoder_path, GraphOptimizationLevel::Level3, decoder_ep)?;
        let decoder_params = extract_decoder_params(&decoder_session)?;

        Ok((
            ActiveBackend::Other {
                decoder: Mutex::new(decoder_session),
            },
            decoder_params,
        ))
    }

    fn make_ep(
        _cuda_graph: bool,
    ) -> ort::execution_providers::ExecutionProviderDispatch {
        #[cfg(target_os = "windows")]
        {
            if _cuda_graph {
                return ort::execution_providers::CUDAExecutionProvider::default()
                    .with_cuda_graph(true)
                    .build();
            }
            ort::execution_providers::CUDAExecutionProvider::default().build()
        }
        #[cfg(target_os = "macos")]
        {
            ort::execution_providers::CoreMLExecutionProvider::default().build()
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            ort::execution_providers::CPUExecutionProvider::default().build()
        }
    }

    /// Predict table structure for one image, returning HTML skeleton + cell bboxes.
    pub fn predict_one(&self, image: &DynamicImage) -> Result<TablePrediction, ExtractError> {
        // 1. Preprocess image → [1, 3, 448, 448]
        let pixel_values = preprocess::preprocess(image);

        // 2. Encode → encoder_memory [1, 784, 512] + enc_out_raw [1, 256, 28, 28]
        let (encoder_memory, enc_out_raw) = {
            let pv_tensor: Value = Value::from_array(pixel_values.into_dyn())
                .map_err(|e| ExtractError::Table(format!("pixel_values tensor: {e}")))?
                .into();

            let mut enc = self
                .encoder
                .lock()
                .map_err(|e| ExtractError::Table(format!("encoder lock: {e}")))?;
            let outputs = enc
                .run(ort::inputs!["pixel_values" => pv_tensor])
                .map_err(|e| ExtractError::Table(format!("encoder run: {e}")))?;

            let enc_mem = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Table(format!("encoder_memory extract: {e}")))?
                .to_owned();
            let enc_raw = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Table(format!("enc_out_raw extract: {e}")))?
                .to_owned();
            (enc_mem, enc_raw)
        };

        // 3. Decode → token sequence + cell hidden states (backend-specific)
        let decode_result = match &self.backend {
            #[cfg(target_os = "windows")]
            ActiveBackend::Cuda { decoder, .. } => {
                let enc_mem_flat = encoder_memory
                    .as_slice()
                    .ok_or_else(|| {
                        ExtractError::Table("encoder_memory not contiguous".into())
                    })?;

                let mut state = decoder
                    .lock()
                    .map_err(|e| ExtractError::Table(format!("decoder lock: {e}")))?;
                cuda::decode_loop(&mut state, enc_mem_flat)?
            }
            ActiveBackend::Other { decoder } => {
                other::decode_loop(decoder, encoder_memory.view(), &self.decoder_params)?
            }
        };

        tracing::debug!(
            "TableFormer: {} tokens, {} cell hidden states",
            decode_result.tokens.len(),
            decode_result.cell_hidden_states.len(),
        );

        // 4. BBox prediction → per-cell bboxes
        let cell_bboxes = if !decode_result.cell_hidden_states.is_empty() {
            let n_cells = decode_result.cell_hidden_states.len();
            let d_model = decode_result.cell_hidden_states[0].len();

            // Stack hidden states → [N, d_model]
            let mut cell_hidden = Array2::<f32>::zeros([n_cells, d_model]);
            for (i, h) in decode_result.cell_hidden_states.iter().enumerate() {
                cell_hidden.row_mut(i).assign(&ndarray::ArrayView1::from(h));
            }

            let enc_raw_tensor: Value = Value::from_array(enc_out_raw.to_owned())
                .map_err(|e| ExtractError::Table(format!("enc_out_raw tensor: {e}")))?
                .into();
            let cell_hidden_tensor: Value = Value::from_array(cell_hidden.into_dyn())
                .map_err(|e| ExtractError::Table(format!("cell_hidden tensor: {e}")))?
                .into();

            let mut bbox_dec = self
                .bbox_decoder
                .lock()
                .map_err(|e| ExtractError::Table(format!("bbox_decoder lock: {e}")))?;

            let outputs = bbox_dec
                .run(ort::inputs![
                    "enc_out_raw" => enc_raw_tensor,
                    "cell_hidden_states" => cell_hidden_tensor,
                ])
                .map_err(|e| ExtractError::Table(format!("bbox_decoder run: {e}")))?;

            let bboxes_raw = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Table(format!("bboxes extract: {e}")))?;
            tracing::debug!(
                "TableFormer: {} bboxes predicted (shape {:?})",
                bboxes_raw.shape()[0],
                bboxes_raw.shape(),
            );

            // Convert cxcywh → xyxy (normalized 0-1)
            let n = bboxes_raw.shape()[0];
            let mut xyxy_bboxes: Vec<[f32; 4]> = Vec::with_capacity(n);
            for i in 0..n {
                let cx = bboxes_raw[[i, 0]];
                let cy = bboxes_raw[[i, 1]];
                let w = bboxes_raw[[i, 2]];
                let h = bboxes_raw[[i, 3]];
                xyxy_bboxes.push([
                    (cx - w / 2.0).clamp(0.0, 1.0),
                    (cy - h / 2.0).clamp(0.0, 1.0),
                    (cx + w / 2.0).clamp(0.0, 1.0),
                    (cy + h / 2.0).clamp(0.0, 1.0),
                ]);
            }

            // Assign bboxes to HTML cells
            let assigned = otsl::assign_cell_bboxes(
                &decode_result.tokens,
                &xyxy_bboxes,
                &decode_result.bboxes_to_merge,
            );

            assigned
        } else {
            Vec::new()
        };

        // 5. Convert token sequence → HTML skeleton
        let html = otsl::otsl_to_html(&decode_result.tokens);
        Ok(TablePrediction { html, cell_bboxes })
    }

    /// Predict table structure for multiple images.
    pub fn predict(&self, images: &[DynamicImage]) -> Result<Vec<TablePrediction>, ExtractError> {
        images.iter().map(|img| self.predict_one(img)).collect()
    }
}

/// Fill empty HTML table cells with text extracted from PDF chars.
///
/// For each cell with a bbox, converts the normalized bbox to PDF-point space,
/// extends edges to include nearby glyphs, extracts text, and inserts it into
/// the HTML.
pub fn fill_table_html(
    html: &str,
    cell_bboxes: &[Option<[f32; 4]>],
    chars: &[PdfChar],
    table_bbox_pt: [f32; 4],
    page_height_pt: f32,
) -> String {
    if cell_bboxes.is_empty() {
        return html.to_string();
    }

    let table_w = table_bbox_pt[2] - table_bbox_pt[0];
    let table_h = table_bbox_pt[3] - table_bbox_pt[1];

    // Pre-convert chars to image-space (Y-down) for bbox matching
    let img_chars: Vec<[f32; 4]> = chars
        .iter()
        .map(|c| {
            // PdfChar bbox is already image space (Y-down) after normalization
            c.bbox
        })
        .collect();

    // Pre-compute all cell bboxes in PDF-point space (Y-down, image-space)
    let all_cells_pt: Vec<Option<[f32; 4]>> = cell_bboxes
        .iter()
        .map(|b| {
            b.map(|bbox| {
                [
                    bbox[0] * table_w + table_bbox_pt[0],
                    bbox[1] * table_h + table_bbox_pt[1],
                    bbox[2] * table_w + table_bbox_pt[0],
                    bbox[3] * table_h + table_bbox_pt[1],
                ]
            })
        })
        .collect();

    // Resolve vertical overlaps between cells in different rows that share
    // column space. The model can produce bboxes where header cells extend
    // into sub-header rows, causing duplicate text extraction.
    let mut all_cells_pt = all_cells_pt;
    resolve_vertical_overlaps(&mut all_cells_pt);

    // Extract text for each cell
    let cell_texts: Vec<String> = all_cells_pt
        .iter()
        .enumerate()
        .map(|(idx, cell_opt)| {
            let Some(cell_pt) = cell_opt else {
                return String::new();
            };

            // Extend bbox edges to snap to nearby glyphs, clamped at neighbor cells.
            let extended = snap_to_nearby_glyphs(*cell_pt, idx, &all_cells_pt, &img_chars);

            text::extract_region_text(
                chars,
                extended,
                page_height_pt,
                &[],
                &[],
                text::AssemblyMode::Reflow,
            )
        })
        .collect();

    // Build filled HTML.
    let mut result = String::with_capacity(
        html.len() + cell_texts.iter().map(|t| t.len()).sum::<usize>(),
    );
    let mut cell_idx = 0;
    let mut i = 0;

    while i < html.len() {
        let rest = &html[i..];

        // Detect opening tag: <td or <th (but not <thead>)
        let is_td = rest.starts_with("<td>") || rest.starts_with("<td ");
        let is_th = rest.starts_with("<th>") || rest.starts_with("<th ");
        if is_td || is_th {
            let is_header = is_th;
            let tag = if is_header { "th" } else { "td" };
            let close_tag = if is_header { "</th>" } else { "</td>" };

            // Find the end of this cell (closing </td> or </th>)
            if let Some(close_pos) = rest.find(close_tag) {
                let cell_html = &rest[..close_pos + close_tag.len()];

                // Emit opening tag with attributes
                let inner_close = rest.find('>').unwrap();
                result.push_str(&rest[..inner_close]);
                result.push('>');

                // Insert cell text
                if cell_idx < cell_texts.len() {
                    let text = &cell_texts[cell_idx];
                    if !text.is_empty() {
                        html_escape_into(text, &mut result);
                    }
                }
                cell_idx += 1;

                result.push_str("</");
                result.push_str(tag);
                result.push('>');
                i += cell_html.len();
                continue;
            }
        }

        // Copy character as-is
        let ch = rest.chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }

    result
}

/// Resolve vertical overlaps between cells in different rows.
///
/// For each pair of cells that overlap both horizontally (share column space)
/// and vertically, clamp the upper cell's bottom edge to the lower cell's top
/// edge. Same-row cells (similar center Y) are skipped.
fn resolve_vertical_overlaps(cells: &mut [Option<[f32; 4]>]) {
    let n = cells.len();
    for i in 0..n {
        let Some(a) = cells[i] else { continue };
        for j in (i + 1)..n {
            let Some(b) = cells[j] else { continue };

            // Check horizontal overlap (x ranges intersect)
            if a[2] <= b[0] || b[2] <= a[0] {
                continue;
            }

            // Check vertical overlap (y ranges intersect)
            if a[3] <= b[1] || b[3] <= a[1] {
                continue;
            }

            // Skip same-row cells (centers at ~same Y)
            let a_cy = (a[1] + a[3]) * 0.5;
            let b_cy = (b[1] + b[3]) * 0.5;
            let min_h = (a[3] - a[1]).min(b[3] - b[1]);
            if (a_cy - b_cy).abs() < min_h * 0.3 {
                continue;
            }

            // Clamp: upper cell's bottom = lower cell's top
            if a_cy < b_cy {
                cells[i] = Some([a[0], a[1], a[2], b[1]]);
            } else {
                cells[j] = Some([b[0], b[1], b[2], a[1]]);
            }
        }
    }
}

/// Extend a cell bbox horizontally to include nearby glyphs just outside left/right edges.
///
/// Iterates until convergence to chain-snap adjacent glyphs. Expansion is
/// clamped at neighboring cell boundaries to prevent bleeding into adjacent cells.
fn snap_to_nearby_glyphs(
    cell: [f32; 4],
    cell_idx: usize,
    all_cells_pt: &[Option<[f32; 4]>],
    char_bboxes: &[[f32; 4]],
) -> [f32; 4] {
    let [mut x1, y1, mut x2, y2] = cell;

    // Compute expansion limits from neighboring cells with vertical overlap.
    let mut left_limit = f32::NEG_INFINITY;
    let mut right_limit = f32::INFINITY;
    for (j, other) in all_cells_pt.iter().enumerate() {
        if j == cell_idx {
            continue;
        }
        let Some(other) = other else { continue };
        // Only consider cells with vertical overlap
        if other[3] > y1 && other[1] < y2 {
            // Neighbor is to the left — its right edge limits our left expansion
            if other[2] <= cell[0] {
                left_limit = left_limit.max(other[2]);
            }
            // Neighbor is to the right — its left edge limits our right expansion
            if other[0] >= cell[2] {
                right_limit = right_limit.min(other[0]);
            }
        }
    }

    loop {
        let (ox1, ox2) = (x1, x2);

        for &[cx1, cy1, cx2, cy2] in char_bboxes {
            let char_cx = (cx1 + cx2) * 0.5;
            let char_cy = (cy1 + cy2) * 0.5;
            let char_w = (cx2 - cx1).abs().max(0.5);

            // Only snap chars whose vertical center is inside the cell
            if char_cy >= y1 && char_cy <= y2 {
                // Just left of cell — don't cross left_limit
                if char_cx < x1 && char_cx >= x1 - char_w && cx1 >= left_limit {
                    x1 = cx1;
                }
                // Just right of cell — don't cross right_limit
                if char_cx > x2 && char_cx <= x2 + char_w && cx2 <= right_limit {
                    x2 = cx2;
                }
            }
        }

        if x1 == ox1 && x2 == ox2 {
            break;
        }
    }

    [x1, y1, x2, y2]
}

/// HTML-escape text into a string buffer.
fn html_escape_into(text: &str, out: &mut String) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
}

/// Build an ORT session with the given execution provider.
fn build_session(
    path: &Path,
    opt_level: GraphOptimizationLevel,
    ep: ort::execution_providers::ExecutionProviderDispatch,
) -> Result<Session, ExtractError> {
    Session::builder()
        .map_err(|e| ExtractError::Model(format!("session builder: {e}")))?
        .with_optimization_level(opt_level)
        .map_err(|e| ExtractError::Model(format!("opt level: {e}")))?
        .with_execution_providers([ep])
        .map_err(|e| ExtractError::Model(format!("EP: {e}")))?
        .commit_from_file(path)
        .map_err(|e| ExtractError::Model(format!("load {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── snap_to_nearby_glyphs ───────────────────────────────────────

    #[test]
    fn snap_extends_to_nearby_glyph() {
        let cell = [10.0, 0.0, 20.0, 5.0];
        let all_cells = vec![Some(cell)];
        // Glyph just to the left of cell, vertical center inside
        let chars = vec![[8.0, 1.0, 9.5, 4.0]];
        let result = snap_to_nearby_glyphs(cell, 0, &all_cells, &chars);
        assert_eq!(result[0], 8.0); // x1 extended left
        assert_eq!(result[2], 20.0); // x2 unchanged
    }

    #[test]
    fn snap_extends_right() {
        let cell = [10.0, 0.0, 20.0, 5.0];
        let all_cells = vec![Some(cell)];
        let chars = vec![[20.5, 1.0, 22.0, 4.0]];
        let result = snap_to_nearby_glyphs(cell, 0, &all_cells, &chars);
        assert_eq!(result[0], 10.0); // x1 unchanged
        assert_eq!(result[2], 22.0); // x2 extended right
    }

    #[test]
    fn snap_blocked_by_neighbor() {
        let cell = [10.0, 0.0, 20.0, 5.0];
        let neighbor = [5.0, 0.0, 9.0, 5.0]; // left neighbor, right edge at 9.0
        let all_cells = vec![Some(cell), Some(neighbor)];
        // Glyph at x=7.0, inside neighbor territory
        let chars = vec![[7.0, 1.0, 8.5, 4.0]];
        let result = snap_to_nearby_glyphs(cell, 0, &all_cells, &chars);
        // Should NOT extend past neighbor's right edge (9.0)
        assert!(result[0] >= 9.0);
    }

    #[test]
    fn snap_chain_extends_iteratively() {
        let cell = [10.0, 0.0, 20.0, 5.0];
        let all_cells = vec![Some(cell)];
        // Two glyphs chained to the left: glyph A just left of cell, glyph B just left of A
        let chars = vec![
            [8.0, 1.0, 9.5, 4.0],  // just left of cell (within 1 char_w)
            [6.0, 1.0, 7.5, 4.0],  // just left of glyph A
        ];
        let result = snap_to_nearby_glyphs(cell, 0, &all_cells, &chars);
        assert_eq!(result[0], 6.0); // chain-snapped to glyph B
    }

    #[test]
    fn snap_ignores_glyph_outside_vertical_range() {
        let cell = [10.0, 0.0, 20.0, 5.0];
        let all_cells = vec![Some(cell)];
        // Glyph with vertical center at 7.0, outside cell Y range [0, 5]
        let chars = vec![[8.0, 6.0, 9.5, 8.0]];
        let result = snap_to_nearby_glyphs(cell, 0, &all_cells, &chars);
        assert_eq!(result[0], 10.0); // unchanged
    }

    // ── resolve_vertical_overlaps ─────────────────────────────────────

    #[test]
    fn overlap_clamps_upper_cell() {
        // Header cell extends into sub-header row
        let mut cells = vec![
            Some([10.0, 0.0, 50.0, 12.0]),  // header, y extends to 12
            Some([10.0, 8.0, 50.0, 20.0]),  // sub-header, starts at 8
        ];
        resolve_vertical_overlaps(&mut cells);
        // Upper cell's y2 should be clamped to lower cell's y1
        assert_eq!(cells[0].unwrap()[3], 8.0);
        // Lower cell unchanged
        assert_eq!(cells[1].unwrap()[1], 8.0);
    }

    #[test]
    fn overlap_skips_side_by_side() {
        // Two cells in the same row, no horizontal overlap
        let mut cells = vec![
            Some([10.0, 0.0, 30.0, 10.0]),
            Some([35.0, 0.0, 55.0, 10.0]),
        ];
        let before = cells.clone();
        resolve_vertical_overlaps(&mut cells);
        assert_eq!(cells, before);
    }

    #[test]
    fn overlap_skips_same_row() {
        // Two cells with same center Y but overlapping X (e.g. colspan)
        let mut cells = vec![
            Some([10.0, 0.0, 30.0, 10.0]),
            Some([20.0, 0.0, 40.0, 10.0]),
        ];
        let before = cells.clone();
        resolve_vertical_overlaps(&mut cells);
        assert_eq!(cells, before);
    }

    #[test]
    fn overlap_skips_non_overlapping_rows() {
        // Two cells with horizontal overlap but no vertical overlap
        let mut cells = vec![
            Some([10.0, 0.0, 50.0, 8.0]),
            Some([10.0, 10.0, 50.0, 20.0]),
        ];
        let before = cells.clone();
        resolve_vertical_overlaps(&mut cells);
        assert_eq!(cells, before);
    }
}
