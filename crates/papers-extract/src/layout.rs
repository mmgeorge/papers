use std::cmp::Ordering;
use std::path::Path;
use std::sync::Mutex;

use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::error::ExtractError;
use crate::types::RegionKind;

/// Class ID → label mapping for PP-DocLayoutV3 (25 classes).
const ID2LABEL: [&str; 25] = [
    "abstract",          // 0
    "algorithm",         // 1
    "aside_text",        // 2
    "chart",             // 3
    "content",           // 4
    "display_formula",   // 5
    "doc_title",         // 6
    "figure_title",      // 7
    "footer",            // 8
    "footer_image",      // 9
    "footnote",          // 10
    "formula_number",    // 11
    "header",            // 12
    "header_image",      // 13
    "image",             // 14
    "inline_formula",    // 15
    "number",            // 16
    "paragraph_title",   // 17
    "reference",         // 18
    "reference_content", // 19
    "seal",              // 20
    "table",             // 21
    "text",              // 22
    "vertical_text",     // 23
    "vision_footnote",   // 24
];

/// A region detected by direct ONNX inference on the layout model.
#[derive(Debug, Clone)]
pub struct DetectedRegion {
    pub kind: RegionKind,
    /// Bounding box in original pixel coordinates: [x1, y1, x2, y2] (image space, Y-down).
    pub bbox_px: [f32; 4],
    pub confidence: f32,
    /// Sort ascending for reading order (from model output column 6).
    pub order_key: f32,
}

/// Direct ONNX layout detector — bypasses oar-ocr's buggy postprocessing.
pub struct LayoutDetector {
    session: Mutex<Session>,
}

impl LayoutDetector {
    /// Create a new detector from the pp-doclayoutv3.onnx model file.
    pub fn new(model_path: &Path) -> Result<Self, ExtractError> {
        let session = Session::builder()
            .map_err(|e| ExtractError::Model(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ExtractError::Model(format!("Failed to set optimization level: {e}")))?
            .with_execution_providers([
                #[cfg(target_os = "windows")]
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                #[cfg(target_os = "macos")]
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| ExtractError::Model(format!("Failed to set execution providers: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                ExtractError::Model(format!("Failed to load layout model: {e}"))
            })?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run layout detection on an image, returning detected regions sorted by reading order.
    pub fn detect(
        &self,
        image: &DynamicImage,
        threshold: f32,
    ) -> Result<Vec<DetectedRegion>, ExtractError> {
        let results = self.detect_batch(&[image], threshold)?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Run batched layout detection on multiple images in a single inference call.
    ///
    /// Uses the model's dynamic batch dimension to process N images at once,
    /// reducing per-image GPU kernel launch overhead. The second model output
    /// (`fetch_name_1`, i32 batch indices) maps each detection to its source image.
    pub fn detect_batch(
        &self,
        images: &[&DynamicImage],
        threshold: f32,
    ) -> Result<Vec<Vec<DetectedRegion>>, ExtractError> {
        let n = images.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Preprocess all images and collect original dimensions
        let mut all_data = Vec::with_capacity(n * 3 * 800 * 800);
        let mut orig_dims: Vec<(f32, f32)> = Vec::with_capacity(n);
        let mut scale_data = Vec::with_capacity(n * 2);
        let mut shape_data = Vec::with_capacity(n * 2);

        for image in images {
            let (tensor, orig_h, orig_w) = preprocess(image);
            all_data.extend_from_slice(tensor.as_slice().unwrap());
            orig_dims.push((orig_w, orig_h));
            scale_data.push(800.0 / orig_h);
            scale_data.push(800.0 / orig_w);
            shape_data.push(800.0f32);
            shape_data.push(800.0f32);
        }

        // Build batched tensors: [N, 3, 800, 800], [N, 2], [N, 2]
        let image_tensor = ndarray::ArrayD::from_shape_vec(
            vec![n, 3, 800, 800],
            all_data,
        ).map_err(|e| ExtractError::Layout(format!("batch image tensor: {e}")))?;
        let scale_tensor = ndarray::ArrayD::from_shape_vec(
            vec![n, 2],
            scale_data,
        ).map_err(|e| ExtractError::Layout(format!("batch scale tensor: {e}")))?;
        let shape_tensor = ndarray::ArrayD::from_shape_vec(
            vec![n, 2],
            shape_data,
        ).map_err(|e| ExtractError::Layout(format!("batch shape tensor: {e}")))?;

        let image_input = TensorRef::from_array_view(&image_tensor)
            .map_err(|e| ExtractError::Layout(format!("Failed to create image tensor: {e}")))?;
        let scale_input = TensorRef::from_array_view(&scale_tensor)
            .map_err(|e| ExtractError::Layout(format!("Failed to create scale tensor: {e}")))?;
        let shape_input = TensorRef::from_array_view(&shape_tensor)
            .map_err(|e| ExtractError::Layout(format!("Failed to create shape tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| ExtractError::Layout(format!("Session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                "image" => image_input,
                "scale_factor" => scale_input,
                "im_shape" => shape_input,
            ])
            .map_err(|e| ExtractError::Layout(format!("Layout inference failed: {e}")))?;

        // fetch_name_0: [total_detections, 7] — boxes for all images
        let boxes = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| ExtractError::Layout(format!("Failed to extract boxes: {e}")))?;
        // fetch_name_1: [total_detections] i32 — batch index per detection
        let batch_idx_output = outputs[1]
            .try_extract_array::<i32>()
            .map_err(|e| ExtractError::Layout(format!("Failed to extract batch indices: {e}")))?;

        // fetch_name_1 is NmsRasterizeNum: [batch_size] i32 — count of valid
        // detections per image. Boxes are [total_rows, 7] laid out as max_per_image
        // rows per batch image (padded with score=0 for unused slots).
        let nms_counts = batch_idx_output;


        // Split detections by batch image using nms_counts.
        // Boxes layout: [N * max_per_image, 7] where each image gets max_per_image
        // rows (padded with score=0). nms_counts[i] = number of valid detections
        // for image i. Valid rows for image i start at i * max_per_image.
        let mut results: Vec<Vec<DetectedRegion>> = (0..n).map(|_| Vec::new()).collect();
        let boxes_shape = boxes.shape();
        if boxes_shape.len() == 2 && boxes_shape[1] == 7 {
            let total_rows = boxes_shape[0];
            let max_per_image = if n > 0 { total_rows / n } else { total_rows };
            let counts = nms_counts.as_slice()
                .ok_or_else(|| ExtractError::Layout("nms counts not contiguous".into()))?;

            for bi in 0..n {
                let valid = counts.get(bi).copied().unwrap_or(0).max(0) as usize;
                let start = bi * max_per_image;
                let (orig_w, orig_h) = orig_dims[bi];

                for i in start..start + valid.min(max_per_image) {
                    if i >= total_rows {
                        break;
                    }
                    let class_id = boxes[[i, 0]] as usize;
                    let score = boxes[[i, 1]];

                    if score < threshold || class_id >= ID2LABEL.len() {
                        continue;
                    }
                    let label = ID2LABEL[class_id];
                    let kind = match RegionKind::from_label(label) {
                        Some(k) => k,
                        None => continue,
                    };
                    results[bi].push(DetectedRegion {
                        kind,
                        bbox_px: [
                            boxes[[i, 2]].clamp(0.0, orig_w),
                            boxes[[i, 3]].clamp(0.0, orig_h),
                            boxes[[i, 4]].clamp(0.0, orig_w),
                            boxes[[i, 5]].clamp(0.0, orig_h),
                        ],
                        confidence: score,
                        order_key: boxes[[i, 6]],
                    });
                }
            }
        }

        // Sort each image's detections by reading order
        for regions in &mut results {
            regions.sort_by(|a, b| {
                a.order_key
                    .partial_cmp(&b.order_key)
                    .unwrap_or(Ordering::Equal)
            });
        }

        Ok(results)
    }
}

/// Preprocess an image for the PP-DocLayoutV3 model.
///
/// Resizes to 800x800 (no aspect ratio preservation), converts to NCHW float32 / 255.0.
fn preprocess(image: &DynamicImage) -> (Array4<f32>, f32, f32) {
    let orig_w = image.width() as f32;
    let orig_h = image.height() as f32;

    let resized = image.resize_exact(800, 800, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();

    let mut data = vec![0.0f32; 3 * 800 * 800];
    for y in 0..800u32 {
        for x in 0..800u32 {
            let pixel = rgb.get_pixel(x, y);
            let idx = (y * 800 + x) as usize;
            data[idx] = pixel[0] as f32 / 255.0;
            data[640_000 + idx] = pixel[1] as f32 / 255.0;
            data[2 * 640_000 + idx] = pixel[2] as f32 / 255.0;
        }
    }

    let tensor = Array4::from_shape_vec((1, 3, 800, 800), data)
        .expect("preprocess: shape mismatch for 1x3x800x800 tensor");
    (tensor, orig_h, orig_w)
}

/// Parse model output into DetectedRegion values.
///
/// Handles both 2D `[num_boxes, 7]` and 4D `[1, num_boxes, 1, 7]` output shapes.
/// Each row: `[class_id, score, x1, y1, x2, y2, order_key]`.
fn postprocess(
    output: &ndarray::ArrayViewD<'_, f32>,
    orig_w: f32,
    orig_h: f32,
    threshold: f32,
) -> Vec<DetectedRegion> {
    let shape = output.shape();

    // Determine number of boxes and stride based on shape
    let (num_boxes, get_row): (usize, Box<dyn Fn(usize) -> [f32; 7]>) = if shape.len() == 2 {
        // [num_boxes, 7]
        let n = shape[0];
        let view = output.clone();
        (
            n,
            Box::new(move |i| {
                let mut row = [0.0f32; 7];
                for j in 0..7 {
                    row[j] = view[[i, j]];
                }
                row
            }),
        )
    } else if shape.len() == 4 {
        // [1, num_boxes, 1, 7]
        let n = shape[1];
        let view = output.clone();
        (
            n,
            Box::new(move |i| {
                let mut row = [0.0f32; 7];
                for j in 0..7 {
                    row[j] = view[[0, i, 0, j]];
                }
                row
            }),
        )
    } else {
        return Vec::new();
    };

    let mut regions = Vec::new();
    for box_idx in 0..num_boxes {
        let row = get_row(box_idx);
        let class_id = row[0] as usize;
        let score = row[1];

        if score < threshold || class_id >= ID2LABEL.len() {
            continue;
        }

        let label = ID2LABEL[class_id];
        let kind = match RegionKind::from_label(label) {
            Some(k) => k,
            None => continue,
        };

        regions.push(DetectedRegion {
            kind,
            bbox_px: [
                row[2].clamp(0.0, orig_w),
                row[3].clamp(0.0, orig_h),
                row[4].clamp(0.0, orig_w),
                row[5].clamp(0.0, orig_h),
            ],
            confidence: score,
            order_key: row[6],
        });
    }

    regions.sort_by(|a, b| {
        a.order_key
            .partial_cmp(&b.order_key)
            .unwrap_or(Ordering::Equal)
    });

    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_shape() {
        let img = DynamicImage::new_rgb8(612, 792);
        let (tensor, orig_h, orig_w) = preprocess(&img);
        assert_eq!(tensor.shape(), &[1, 3, 800, 800]);
        assert!((orig_h - 792.0).abs() < 0.01);
        assert!((orig_w - 612.0).abs() < 0.01);
    }

    #[test]
    fn test_postprocess_2d() {
        // Simulate a 2D output: [2, 7]
        let data = vec![
            // class_id=22(text), score=0.95, x1=10, y1=20, x2=100, y2=50, order=1.0
            22.0, 0.95, 10.0, 20.0, 100.0, 50.0, 1.0,
            // class_id=6(doc_title), score=0.20, x1=10, y1=5, x2=100, y2=15, order=0.0
            // Below threshold 0.3
            6.0, 0.20, 10.0, 5.0, 100.0, 15.0, 0.0,
        ];
        let arr = ndarray::ArrayD::from_shape_vec(vec![2, 7], data).unwrap();
        let regions = postprocess(&arr.view(), 200.0, 300.0, 0.3);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].kind, RegionKind::Text);
        assert!((regions[0].confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_postprocess_4d() {
        // Simulate a 4D output: [1, 2, 1, 7]
        let data = vec![
            // Box 0: class_id=6(doc_title), score=0.9, order=0.0
            6.0, 0.9, 50.0, 10.0, 300.0, 60.0, 0.0,
            // Box 1: class_id=22(text), score=0.8, order=1.0
            22.0, 0.8, 50.0, 70.0, 300.0, 200.0, 1.0,
        ];
        let arr = ndarray::ArrayD::from_shape_vec(vec![1, 2, 1, 7], data).unwrap();
        let regions = postprocess(&arr.view(), 400.0, 600.0, 0.3);
        assert_eq!(regions.len(), 2);
        // Sorted by order_key
        assert_eq!(regions[0].kind, RegionKind::Title);
        assert_eq!(regions[1].kind, RegionKind::Text);
    }

    #[test]
    fn test_postprocess_clamps_coords() {
        let data = vec![
            22.0, 0.9, -10.0, -5.0, 999.0, 888.0, 0.0,
        ];
        let arr = ndarray::ArrayD::from_shape_vec(vec![1, 7], data).unwrap();
        let regions = postprocess(&arr.view(), 200.0, 300.0, 0.3);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].bbox_px, [0.0, 0.0, 200.0, 300.0]);
    }

    #[test]
    fn test_postprocess_reading_order() {
        let data = vec![
            22.0, 0.9, 10.0, 10.0, 100.0, 50.0, 3.0, // order_key = 3
            22.0, 0.8, 10.0, 60.0, 100.0, 100.0, 1.0, // order_key = 1
            6.0, 0.95, 10.0, 5.0, 200.0, 30.0, 0.0,  // order_key = 0
        ];
        let arr = ndarray::ArrayD::from_shape_vec(vec![3, 7], data).unwrap();
        let regions = postprocess(&arr.view(), 400.0, 400.0, 0.3);
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].kind, RegionKind::Title); // order_key 0
        assert_eq!(regions[1].kind, RegionKind::Text); // order_key 1
        assert_eq!(regions[2].kind, RegionKind::Text); // order_key 3
    }

    #[test]
    fn test_id2label_coverage() {
        // Verify all 25 labels map to a RegionKind
        for (id, label) in ID2LABEL.iter().enumerate() {
            assert!(
                RegionKind::from_label(label).is_some(),
                "ID2LABEL[{id}] = {label:?} has no RegionKind mapping"
            );
        }
    }
}
