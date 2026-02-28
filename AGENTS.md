# Project-Wide Agent Instructions

## Temporary files

All temporary or scratch files (test outputs, diagnostic dumps, intermediate data,
etc.) **must** go in the `.temp/` directory at the project root. Never write temp
files to `data/`, source directories, or the system temp directory.

```bash
# correct
.temp/vbd_p4_output.txt
.temp/debug_chars.json

# wrong
data/debug_output.txt
crates/papers-extract/scratch.txt
/tmp/papers-test.json
```

The `.temp/` directory is gitignored. Create it on demand if it doesn't exist.
