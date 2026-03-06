//! Image preprocessing for TableFormer: resize + normalize matching docling.

use image::DynamicImage;
use ndarray::Array4;

/// Target image size (square).
const IMAGE_SIZE: u32 = 448;

/// Dataset-specific normalization (from tm_config.json, BGR channel order).
const MEAN: [f32; 3] = [0.94247851, 0.94254675, 0.94292611];
const STD: [f32; 3] = [0.17910956, 0.17940403, 0.17931663];

/// Preprocess image for TableFormer encoder input.
///
/// Matches `tf_predictor._prepare_image()`:
/// - BGR channel order (OpenCV convention)
/// - Normalize: `(pixel/255 - mean) / std`
/// - transpose(2,1,0): HWC → CWH (docling convention, spatial transpose)
///
/// Returns `[1, 3, 448, 448]` f32.
pub fn preprocess(image: &DynamicImage) -> Array4<f32> {
    let resized = image.resize_exact(IMAGE_SIZE, IMAGE_SIZE, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let mut result = Array4::<f32>::zeros([1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize]);

    for y in 0..IMAGE_SIZE as usize {
        for x in 0..IMAGE_SIZE as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // RGB → BGR + normalize + spatial transpose (swap x,y in output)
            // docling: result[c, x, y] = normalize(bgr_pixel[h=y, w=x, c])
            // With transpose(2,1,0) on HWC: output[c, w, h] = input[h, w, c]
            // So output[c, x, y] = normalized_input[y, x, c]  (swap spatial dims)
            result[[0, 0, x, y]] = (b - MEAN[0]) / STD[0]; // B channel
            result[[0, 1, x, y]] = (g - MEAN[1]) / STD[1]; // G channel
            result[[0, 2, x, y]] = (r - MEAN[2]) / STD[2]; // R channel
        }
    }

    result
}
