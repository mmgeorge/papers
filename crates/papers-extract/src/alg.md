# Follow-up: Algorithm detection improvements

## Current state
- `code_score()` from output.rs is used but triggers false positives on math notation
- Math markers (`$`, `{`, `}`, math italic Unicode) now suppress code_score entirely
- Single/two-line blocks are excluded from Algorithm classification
- Result: zero false Algorithm blocks in avbd, but real code blocks in programming books may also be missed

## Better approach (to implement)
Replace `code_score()` heuristic with targeted algorithm detection:

1. **Caption label**: blocks starting with "Algorithm N" → already caught by `match_label_prefix()` as FigureTitle
2. **Line numbers**: blocks where lines start with sequential numbers (1, 2, 3...) or keywords like "Step 1:", "Input:", "Output:", "Require:", "Ensure:"
3. **Pseudocode keywords**: `while`, `for`, `if`, `then`, `do`, `end`, `return`, `repeat`, `until` combined with indentation structure
4. **Actual code**: `code_score()` should still be used for real programming language code (C/Python/GLSL etc.) but NOT when the block contains math markers (`$`, math italic Unicode, Greek letters)

## Files to modify
- `text_only.rs` — classification logic in `extract_page_text_blocks()`
- `text_cleanup.rs` — potential shared `is_likely_algorithm()` function
- `output.rs` — `code_score()` may need a companion `algorithm_score()` for pseudocode

## Test cases
- avbd/vbd: equations should NOT be Algorithm
- compilers: code listings SHOULD be Algorithm
- shaders: GLSL code SHOULD be Algorithm
- calculus: SIR Python program SHOULD be Algorithm
