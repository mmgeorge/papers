//! Debug: find chars with unusual properties (empty font, small size, etc)
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pdf_path = PathBuf::from(args.get(1).expect("pdf_path required"));
    let page_num: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(68);
    let page_idx = page_num - 1;

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium.load_pdf_from_file(&pdf_path, None).expect("Failed to load PDF");
    let page = doc.pages().get(page_idx as u16).expect("Failed to get page");
    
    let mut chars = papers_extract::pdf::extract_page_chars(&page, page_idx)
        .expect("Failed to extract chars");
    papers_extract::pdf::normalize_chars_to_image_space(&mut chars, page.height().value);
    
    // Show font size distribution for j chars vs non-j chars
    let j_sizes: Vec<f32> = chars.iter().filter(|c| c.codepoint == 'j').map(|c| c.font_size).collect();
    let other_sizes: Vec<f32> = chars.iter().filter(|c| c.codepoint != 'j' && c.font_size > 0.0).map(|c| c.font_size).take(20).collect();
    
    println!("j char font sizes (first 20): {:?}", &j_sizes[..j_sizes.len().min(20)]);
    println!("other char font sizes (first 20): {:?}", &other_sizes[..other_sizes.len().min(20)]);
    
    // Show all unique font sizes on page
    let mut all_sizes: Vec<f32> = chars.iter().map(|c| c.font_size).collect();
    all_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_sizes.dedup();
    println!("\nAll unique font sizes: {:?}", &all_sizes[..all_sizes.len().min(30)]);
    
    // Show unique font names
    let mut font_names: Vec<String> = chars.iter().map(|c| c.font_name.clone()).collect();
    font_names.sort();
    font_names.dedup();
    println!("\nAll unique font names: {:?}", &font_names[..font_names.len().min(20)]);
    
    // Show context around j chars (2140-2175)
    println!("\nChars near first j (indices 2135-2175):");
    for (i, c) in chars.iter().enumerate() {
        if (2135..=2175).contains(&i) {
            println!("  {} '{}' size={:.2} font={:?}", i, 
                if c.codepoint.is_control() { '?' } else { c.codepoint },
                c.font_size, c.font_name);
        }
    }
}
