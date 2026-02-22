use std::collections::HashMap;

pub struct SamplingContext<'a> {
    pub generated_tokens: &'a [u32],
    pub prompt_tokens: &'a [u32],
    pub token_counts: &'a HashMap<u32, usize>,
}

pub struct SampleResult {
    pub token_id: u32,
    pub logprob: f32,
}
