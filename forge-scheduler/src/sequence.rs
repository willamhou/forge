use forge_core::SamplingParams;

/// The lifecycle state of a sequence in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    /// Waiting in the queue for initial prefill.
    Waiting,
    /// Currently being processed (prefill or decode).
    Running,
    /// Completed (EOS, max tokens, cancelled).
    Finished,
}

/// Full state of a sequence tracked by the scheduler.
#[derive(Debug)]
pub struct SequenceState {
    pub seq_id: u64,
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub state: SeqState,
    pub is_prefilled: bool,
    /// How many prompt tokens have already been prefilled (for chunked prefill).
    pub prefill_offset: usize,
}

impl SequenceState {
    pub fn new(
        seq_id: u64,
        request_id: String,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            seq_id,
            request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            sampling_params,
            state: SeqState::Waiting,
            is_prefilled: false,
            prefill_offset: 0,
        }
    }

    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
}
