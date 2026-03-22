# Resolved: Formula left-clipping issue

## Problem (resolved)
Display formulas like `f_i = -1/Δt² M_i(x_i - y_i) + Σ f_ij (5)` had their
left-hand side (`f_i`) clipped. The formula chars spanned multiple Y levels
(numerator, main line, denominator), causing `group_into_lines()` to split
them and `group_into_blocks()` to fragment the formula into multiple blocks.

## Fix (implemented)
Three-part fix:

1. **Formula line expansion** — after initial formula line detection, adjacent
   narrow lines with math/garbled chars are expanded into the formula group.
   This keeps fraction parts (numerator "1", denominator "Δt²") with the
   main formula line during block grouping.

2. **Fragment merging** — after Pass 1 identifies formula blocks, small
   adjacent blocks with math content are merged into the formula bbox.

3. **Dedup merging** — `dedup_overlapping_formulas()` now MERGES overlapping
   formulas (expanding bboxes) instead of picking one and discarding the
   other. Requires both vertical AND horizontal overlap to prevent merging
   across columns.
