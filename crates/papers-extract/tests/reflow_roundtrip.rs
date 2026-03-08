use papers_extract::output;
use papers_extract::types::ExtractionResult;
use std::fs;

/// Normalize heading levels: strip leading `#` chars and compare content only.
fn normalize_heading(line: &str) -> String {
    let trimmed = line.trim_start_matches('#').trim_start();
    if line.starts_with('#') {
        format!("# {trimmed}")
    } else {
        line.to_string()
    }
}

#[test]
fn reflow_roundtrip_avbd() {
    let json_path = "../../test-extract/avbd/avbd.json";
    let md_path = "../../test-extract/avbd/avbd.md";

    if !std::path::Path::new(json_path).exists() {
        eprintln!("Skipping: test-extract/avbd/avbd.json not found");
        return;
    }

    let json_str = fs::read_to_string(json_path).expect("read avbd.json");
    let result: ExtractionResult = serde_json::from_str(&json_str).expect("parse avbd.json");

    let doc = output::reflow(&result);

    // Write reflow JSON for inspection
    let reflow_json = serde_json::to_string_pretty(&doc).expect("serialize reflow");
    fs::write("../../.temp/avbd-reflow-test.json", &reflow_json).ok();

    let md = output::render_markdown_from_reflow(&doc);
    fs::write("../../.temp/avbd-reflow-test.md", &md).ok();

    let expected = fs::read_to_string(md_path).expect("read avbd.md");

    // Compare content line-by-line, normalizing heading levels
    let md_lines: Vec<String> = md.lines().map(normalize_heading).collect();
    let exp_lines: Vec<String> = expected.trim_end().lines().map(normalize_heading).collect();

    for (i, (actual, expected)) in md_lines.iter().zip(exp_lines.iter()).enumerate() {
        if actual != expected {
            panic!(
                "Content diff at line {}:\n  expected: {:?}\n  actual:   {:?}",
                i + 1,
                expected,
                actual
            );
        }
    }
    if md_lines.len() != exp_lines.len() {
        let min = md_lines.len().min(exp_lines.len());
        let extra_actual = if md_lines.len() > min {
            format!("\n  first extra actual: {:?}", &md_lines[min])
        } else {
            String::new()
        };
        let extra_expected = if exp_lines.len() > min {
            format!("\n  first extra expected: {:?}", &exp_lines[min])
        } else {
            String::new()
        };
        panic!(
            "Line count differs: expected {} lines, got {} lines{}{}",
            exp_lines.len(),
            md_lines.len(),
            extra_expected,
            extra_actual,
        );
    }

    eprintln!("Reflow roundtrip test passed (content match, heading depths may differ)");
}
