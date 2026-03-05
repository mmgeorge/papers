//! Image preprocessing for GLM-OCR vision encoder.
//!
//! Pipeline (matches Glm46VImageProcessorFast):
//!   1. Resize to (H, W) where both are multiples of 28
//!   2. Normalize: (pixel/255 - mean) / std
//!   3. Duplicate for temporal_patch_size=2 (same image × 2 frames)
//!   4. Extract patches and flatten to [num_patches, 1176]

use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::{Array2, Array3};

use super::config::*;

/// Smart-resize dimensions to be multiples of GRID_UNIT (28), keeping the
/// total pixel area within [MIN_PIXELS, MAX_PIXELS].
///
/// Matches the exact logic from transformers Glm46VImageProcessor.smart_resize.
pub(crate) fn smart_resize(orig_h: u32, orig_w: u32) -> (u32, u32) {
    let factor = GRID_UNIT as f64;
    let num_frames = TEMPORAL_PATCH_SIZE as f64;

    // If either dimension < factor, scale up to ensure both >= factor
    let (mut height, mut width) = (orig_h as f64, orig_w as f64);
    if height < factor || width < factor {
        let scale = (factor / height).max(factor / width);
        height = (height * scale).floor();
        width = (width * scale).floor();
    }

    // Round to nearest multiple of factor
    let mut h_bar = (height / factor).round().max(1.0) as u32 * GRID_UNIT;
    let mut w_bar = (width / factor).round().max(1.0) as u32 * GRID_UNIT;
    let t_bar = num_frames; // = temporal_patch_size = 2

    // Clamp total area (t_bar * h_bar * w_bar) to [MIN_PIXELS, MAX_PIXELS]
    let total = t_bar * h_bar as f64 * w_bar as f64;
    if total > MAX_PIXELS as f64 {
        let beta = (num_frames * height * width / MAX_PIXELS as f64).sqrt();
        h_bar = (height / beta / factor).floor().max(1.0) as u32 * GRID_UNIT;
        w_bar = (width / beta / factor).floor().max(1.0) as u32 * GRID_UNIT;
    } else if total < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (num_frames * height * width)).sqrt();
        h_bar = (height * beta / factor).ceil() as u32 * GRID_UNIT;
        w_bar = (width * beta / factor).ceil() as u32 * GRID_UNIT;
    }

    (h_bar, w_bar)
}

/// Preprocess a formula image for the GLM-OCR vision encoder.
///
/// Returns:
///   - pixel_values: [num_patches, 1176] f32
///   - grid_thw: [1, 3] i64 = [temporal=1, h_patches, w_patches]
pub(crate) fn preprocess_image(image: &DynamicImage) -> (Array2<f32>, Array2<i64>) {
    let rgb = image.to_rgb8();
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    let (target_h, target_w) = smart_resize(orig_h, orig_w);

    // Resize to exact target dimensions
    // CatmullRom = Bicubic, matching HuggingFace's default resample=3 (PIL.BICUBIC)
    let resized = image.resize_exact(target_w, target_h, FilterType::CatmullRom);
    let resized_rgb = resized.to_rgb8();

    let grid_h = (target_h / PATCH_SIZE) as usize;
    let grid_w = (target_w / PATCH_SIZE) as usize;
    let num_patches = grid_h * grid_w;
    let ps = PATCH_SIZE as usize;
    let sm = SPATIAL_MERGE as usize;
    let merged_h = grid_h / sm;
    let merged_w = grid_w / sm;

    // Build pixel_values: [num_patches, 1176]
    // Patch ordering: [gh, gw, mh, mw] where gh/gw iterate over merged groups
    // and mh/mw (each 0..merge_size-1) iterate within each 2×2 merge group.
    // Memory layout per patch: C × T × H × W (channels outermost)
    // where C=3, T=temporal_patch_size=2, H=patch_size=14, W=patch_size=14
    // Since temporal_patch_size=2 and it's a single image, both frames are identical.
    let spatial = ps * ps; // 196
    let temporal_spatial = TEMPORAL_PATCH_SIZE * spatial; // 392
    let mut pixel_values = vec![0.0f32; num_patches * PATCH_ELEM];

    let mut patch_idx = 0;
    for bh in 0..merged_h {
        for bw in 0..merged_w {
            for mh in 0..sm {
                for mw in 0..sm {
                    let patch_offset = patch_idx * PATCH_ELEM;
                    let y_start = (bh * sm + mh) * ps;
                    let x_start = (bw * sm + mw) * ps;

                    for c in 0..3usize {
                        for t in 0..TEMPORAL_PATCH_SIZE {
                            for py in 0..ps {
                                for px in 0..ps {
                                    let pixel = resized_rgb
                                        .get_pixel((x_start + px) as u32, (y_start + py) as u32);
                                    let val = pixel[c] as f32 / 255.0;
                                    let normalized = (val - NORM_MEAN[c]) / NORM_STD[c];
                                    let idx = patch_offset
                                        + c * temporal_spatial
                                        + t * spatial
                                        + py * ps
                                        + px;
                                    pixel_values[idx] = normalized;
                                }
                            }
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }

    let pv_array = Array2::from_shape_vec([num_patches, PATCH_ELEM], pixel_values)
        .expect("pixel_values shape mismatch");

    // grid_thw: [1, h_patches, w_patches]
    let grid_thw =
        Array2::from_shape_vec([1, 3], vec![1i64, grid_h as i64, grid_w as i64])
            .expect("grid_thw shape");

    (pv_array, grid_thw)
}

// ── Vision position IDs (M-RoPE) ─────────────────────────────────────

/// Compute position IDs for vision rotary embeddings.
/// Returns (pos_ids: [N, 2], max_grid_size: i64).
pub(crate) fn compute_vision_pos_ids(grid_thw: &Array2<i64>) -> (Array2<i64>, i64) {
    let mut pos_ids_list: Vec<Vec<[i64; 2]>> = Vec::new();
    let mut max_grid_size: i64 = 0;

    for row in 0..grid_thw.shape()[0] {
        let t = grid_thw[[row, 0]] as usize;
        let h = grid_thw[[row, 1]] as usize;
        let w = grid_thw[[row, 2]] as usize;
        let sm = SPATIAL_MERGE as usize;

        max_grid_size = max_grid_size.max(h as i64).max(w as i64);

        // Height position IDs
        let mut hpos = vec![0i64; h * w];
        for y in 0..h {
            for x in 0..w {
                hpos[y * w + x] = y as i64;
            }
        }

        // Reshape to [h/sm, sm, w/sm, sm], transpose to [h/sm, w/sm, sm, sm], flatten
        let mh = h / sm;
        let mw = w / sm;
        let mut h_merged = vec![0i64; h * w];
        for bh in 0..mh {
            for bw in 0..mw {
                for sh in 0..sm {
                    for sw in 0..sm {
                        let src_idx = (bh * sm + sh) * w + (bw * sm + sw);
                        let dst_idx = ((bh * mw + bw) * sm + sh) * sm + sw;
                        h_merged[dst_idx] = hpos[src_idx];
                    }
                }
            }
        }

        // Width position IDs
        let mut wpos = vec![0i64; h * w];
        for y in 0..h {
            for x in 0..w {
                wpos[y * w + x] = x as i64;
            }
        }

        let mut w_merged = vec![0i64; h * w];
        for bh in 0..mh {
            for bw in 0..mw {
                for sh in 0..sm {
                    for sw in 0..sm {
                        let src_idx = (bh * sm + sh) * w + (bw * sm + sw);
                        let dst_idx = ((bh * mw + bw) * sm + sh) * sm + sw;
                        w_merged[dst_idx] = wpos[src_idx];
                    }
                }
            }
        }

        // Stack [h, w] pairs and tile over time
        let frame_len = h * w;
        let mut frame_pos: Vec<[i64; 2]> = Vec::with_capacity(frame_len);
        for i in 0..frame_len {
            frame_pos.push([h_merged[i], w_merged[i]]);
        }

        // Tile over t frames
        let mut tiled = Vec::with_capacity(t * frame_len);
        for _ in 0..t {
            tiled.extend_from_slice(&frame_pos);
        }
        pos_ids_list.push(tiled);
    }

    // Concatenate all entries
    let total: usize = pos_ids_list.iter().map(|v| v.len()).sum();
    let mut flat = Vec::with_capacity(total * 2);
    for entry in &pos_ids_list {
        for &[h, w] in entry {
            flat.push(h);
            flat.push(w);
        }
    }

    let pos_ids = Array2::from_shape_vec([total, 2], flat).expect("pos_ids shape");
    (pos_ids, max_grid_size)
}

/// Build 3D M-RoPE position IDs for text+vision tokens.
/// Returns [3, 1, seq_len] position IDs.
pub(crate) fn build_position_ids(input_ids: &[i64], grid_thw: &Array2<i64>) -> Array3<i64> {
    let seq_len = input_ids.len();
    let mut position_ids = vec![0i64; 3 * seq_len];

    let mut pos: i64 = 0;
    let mut i = 0;
    let mut img_idx = 0;

    while i < seq_len {
        if input_ids[i] != IMAGE_TOKEN_ID {
            // Text token: all 3 dims get same position
            position_ids[i] = pos; // dim 0: temporal
            position_ids[seq_len + i] = pos; // dim 1: height
            position_ids[2 * seq_len + i] = pos; // dim 2: width
            pos += 1;
            i += 1;
        } else {
            // Vision tokens: spatial layout
            let t_grid = grid_thw[[img_idx, 0]] as usize;
            let h_grid = grid_thw[[img_idx, 1]] as usize;
            let w_grid = grid_thw[[img_idx, 2]] as usize;
            let merged_h = h_grid / SPATIAL_MERGE as usize;
            let merged_w = w_grid / SPATIAL_MERGE as usize;
            let num_vision_tokens = t_grid * merged_h * merged_w;

            let temporal_pos = pos;
            for vi in 0..num_vision_tokens {
                let row = (vi % (merged_h * merged_w)) / merged_w;
                let col = (vi % (merged_h * merged_w)) % merged_w;
                position_ids[i + vi] = temporal_pos; // dim 0: temporal
                position_ids[seq_len + i + vi] = pos + row as i64; // dim 1: height
                position_ids[2 * seq_len + i + vi] = pos + col as i64; // dim 2: width
            }

            pos += merged_h.max(merged_w) as i64;
            i += num_vision_tokens;
            img_idx += 1;
        }
    }

    Array3::from_shape_vec([3, 1, seq_len], position_ids).expect("position_ids shape")
}
