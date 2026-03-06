//! Shared types and helpers for TableFormer autoregressive decoding.

use ort::session::Session;

use crate::error::ExtractError;

use super::otsl;

/// Parameters extracted from the decoder ONNX session metadata.
pub struct DecoderParams {
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq: usize,
    pub head_dim: usize,
}

/// Result of the autoregressive decode loop.
pub struct DecodeResult {
    /// Full token sequence including START and END.
    pub tokens: Vec<i64>,
    /// Hidden states collected for bbox-bearing tokens.
    pub cell_hidden_states: Vec<Vec<f32>>,
    /// Map from span-start bbox index → span-end bbox index (for lcel merge).
    pub bboxes_to_merge: Vec<(usize, usize)>,
}

/// Extract decoder shape parameters from the ONNX session input metadata.
pub fn extract_decoder_params(session: &Session) -> Result<DecoderParams, ExtractError> {
    let num_layers = session
        .inputs()
        .iter()
        .filter(|inp| {
            inp.name().starts_with("past_key_values.") && inp.name().ends_with(".key")
        })
        .count();

    if num_layers == 0 {
        return Err(ExtractError::Model(
            "decoder has no past_key_values.*.key inputs".into(),
        ));
    }

    let ref_input = session
        .inputs()
        .iter()
        .find(|inp| inp.name() == "past_key_values.0.key")
        .ok_or_else(|| {
            ExtractError::Model("no past_key_values.0.key input in decoder".into())
        })?;

    let shape = ref_input
        .dtype()
        .tensor_shape()
        .ok_or_else(|| ExtractError::Model("past_key_values.0.key is not a tensor".into()))?;

    let dims: Vec<i64> = shape.iter().copied().collect();
    if dims.len() != 4 {
        return Err(ExtractError::Model(format!(
            "past_key_values.0.key has {} dims, expected 4",
            dims.len()
        )));
    }

    Ok(DecoderParams {
        num_layers,
        num_heads: dims[1] as usize,
        max_seq: dims[2] as usize,
        head_dim: dims[3] as usize,
    })
}

/// Tracking state for OTSL structure correction and bbox collection.
///
/// Shared between CPU and CUDA decode loops.
pub(crate) struct TokenTracker {
    pub tokens: Vec<i64>,
    pub cell_hidden_states: Vec<Vec<f32>>,
    pub bboxes_to_merge: Vec<(usize, usize)>,
    skip_next_tag: bool,
    prev_tag_ucel: bool,
    line_num: u32,
    first_lcel: bool,
    cur_bbox_start: Option<usize>,
    bbox_ind: usize,
}

impl TokenTracker {
    pub fn new() -> Self {
        Self {
            tokens: vec![otsl::START],
            cell_hidden_states: Vec::new(),
            bboxes_to_merge: Vec::new(),
            skip_next_tag: true,
            prev_tag_ucel: false,
            line_num: 0,
            first_lcel: true,
            cur_bbox_start: None,
            bbox_ind: 0,
        }
    }

    /// Apply structure correction to the raw argmax token.
    pub fn correct(&self, mut tag: i64) -> i64 {
        if self.line_num == 0 && tag == otsl::XCEL {
            tag = otsl::LCEL;
        }
        if self.prev_tag_ucel && tag == otsl::LCEL {
            tag = otsl::FCEL;
        }
        tag
    }

    /// Process a corrected token, collecting hidden state if needed.
    ///
    /// Returns true if hidden_state should be collected for this token.
    /// Call `push_hidden` afterwards with the actual data.
    pub fn needs_hidden(&self, tag: i64) -> bool {
        let cell_needs = !self.skip_next_tag
            && matches!(
                tag,
                otsl::FCEL
                    | otsl::ECEL
                    | otsl::CHED
                    | otsl::RHED
                    | otsl::SROW
                    | otsl::NL
                    | otsl::UCEL
            );
        let lcel_needs = tag == otsl::LCEL && self.first_lcel;
        cell_needs || lcel_needs
    }

    /// Push a token and its hidden state (if applicable), updating tracking state.
    pub fn push(&mut self, tag: i64, hidden: Option<Vec<f32>>) {
        // Collect hidden states for bbox prediction
        if !self.skip_next_tag {
            if matches!(
                tag,
                otsl::FCEL
                    | otsl::ECEL
                    | otsl::CHED
                    | otsl::RHED
                    | otsl::SROW
                    | otsl::NL
                    | otsl::UCEL
            ) {
                if let Some(h) = &hidden {
                    self.cell_hidden_states.push(h.clone());
                }
                if let Some(start) = self.cur_bbox_start {
                    self.bboxes_to_merge.push((start, self.bbox_ind));
                    self.cur_bbox_start = None;
                }
                self.bbox_ind += 1;
            }
        }

        // Handle horizontal span (lcel) bboxes
        if tag != otsl::LCEL {
            self.first_lcel = true;
        } else if self.first_lcel {
            if let Some(h) = &hidden {
                self.cell_hidden_states.push(h.clone());
            }
            self.first_lcel = false;
            self.cur_bbox_start = Some(self.bbox_ind);
            self.bbox_ind += 1;
        }

        // Update tracking state
        self.skip_next_tag = otsl::is_skip_trigger(tag);
        self.prev_tag_ucel = tag == otsl::UCEL;
        if tag == otsl::NL {
            self.line_num += 1;
        }

        self.tokens.push(tag);
    }

    pub fn into_result(self) -> DecodeResult {
        DecodeResult {
            tokens: self.tokens,
            cell_hidden_states: self.cell_hidden_states,
            bboxes_to_merge: self.bboxes_to_merge,
        }
    }
}

/// Argmax over a slice of f32 logits, returning index as i64.
pub(crate) fn argmax(logits: &[f32]) -> i64 {
    let mut best = 0i64;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            best = i as i64;
        }
    }
    best
}
