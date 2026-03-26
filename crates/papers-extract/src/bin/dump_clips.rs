//! Diagnostic tool: dump clip path info for all page objects.
//! Usage: cargo run --release --bin dump_clips -- data/avbd.pdf [page_number]

use pdfium_render::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pdf_path = args.get(1).expect("Usage: dump_clips <pdf> [page]");
    let page_filter: Option<u32> = args.get(2).and_then(|s| s.parse().ok());
    let dump_all = args.get(3).map(|s| s == "--all").unwrap_or(false);

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("load pdfium");
    let doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .expect("load pdf");

    for page_idx in 0..doc.pages().len() {
        if let Some(pf) = page_filter {
            if page_idx as u32 + 1 != pf {
                continue;
            }
        }
        let page = doc.pages().get(page_idx).unwrap();
        println!("=== Page {} (height={:.1}, width={:.1}) ===",
            page_idx + 1, page.height().value, page.width().value);

        for (obj_idx, obj) in page.objects().iter().enumerate() {
            let obj_type = obj.object_type();
            if !dump_all && obj_type != PdfPageObjectType::Image
                && obj_type != PdfPageObjectType::XObjectForm
            {
                continue;
            }

            let bounds_str = obj.bounds().ok().map(|r| {
                format!("[{:.1}, {:.1}, {:.1}, {:.1}]",
                    r.left().value, r.bottom().value, r.right().value, r.top().value)
            }).unwrap_or("?".into());

            let form_info = if obj_type == PdfPageObjectType::XObjectForm {
                obj.as_x_object_form_object()
                    .map(|f| format!(" children={}", f.len()))
                    .unwrap_or_default()
            } else {
                String::new()
            };

            println!("\n  obj[{}] {:?}{} bounds={}",
                obj_idx, obj_type, form_info, bounds_str);

            // Dump clip path on the object itself
            dump_clip("    self", &obj);

            // For Form XObjects, dump clips on first few children
            if obj_type == PdfPageObjectType::XObjectForm {
                if let Some(form) = obj.as_x_object_form_object() {
                    let check_count = form.len().min(5);
                    for ci in 0..check_count {
                        if let Ok(child) = form.get(ci) {
                            dump_clip(
                                &format!("    child[{}] {:?}", ci, child.object_type()),
                                &child,
                            );
                        }
                    }
                }
            }
        }
    }
}

fn dump_clip(prefix: &str, obj: &PdfPageObject) {
    match obj.get_clip_path() {
        None => println!("{}: clip=None", prefix),
        Some(clip) => {
            let len = clip.len();
            let is_empty = clip.is_empty();
            println!("{}: clip len={} is_empty={}", prefix, len, is_empty);

            if !is_empty && len < 100 {
                for sp_idx in 0..len.min(5) {
                    match clip.get(sp_idx) {
                        Ok(sp) => {
                            let seg_count = sp.len();
                            print!("{}   sub_path[{}]: {} segments", prefix, sp_idx, seg_count);
                            if seg_count > 0 && seg_count < 20 {
                                print!(" [");
                                for si in 0..seg_count {
                                    if let Ok(seg) = sp.get(si) {
                                        let (x, y) = (seg.x().value, seg.y().value);
                                        let st = match seg.segment_type() {
                                            PdfPathSegmentType::MoveTo => "M",
                                            PdfPathSegmentType::LineTo => "L",
                                            PdfPathSegmentType::BezierTo => "B",
                                            _ => "?",
                                        };
                                        print!("{}({:.1},{:.1})", st, x, y);
                                        if si + 1 < seg_count { print!(", "); }
                                    }
                                }
                                print!("]");
                            }
                            println!();
                        }
                        Err(e) => println!("{}   sub_path[{}]: ERROR {:?}", prefix, sp_idx, e),
                    }
                }
            }
        }
    }
}
