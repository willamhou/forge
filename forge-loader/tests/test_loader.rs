use forge_loader::LlamaConfig;

#[test]
fn test_parse_llama_config() {
    let json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    }"#;
    let config: LlamaConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_key_value_heads, 32);
    assert_eq!(config.vocab_size, 32000);
}

#[test]
fn test_config_defaults() {
    let json = r#"{
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "vocab_size": 32000,
        "max_position_embeddings": 2048
    }"#;
    let config: LlamaConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.num_key_value_heads, 32); // default
    assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
    assert!((config.rope_theta - 10000.0).abs() < 1e-6);
    assert!(config.head_dim.is_none());
}

#[test]
fn test_to_model_config() {
    let json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0
    }"#;
    let config: LlamaConfig = serde_json::from_str(json).unwrap();
    let mc = config.to_model_config();
    assert_eq!(mc.head_dim, 128); // 4096 / 32
    assert_eq!(mc.num_key_value_heads, 8);
    assert!((mc.rms_norm_eps - 1e-6).abs() < 1e-10);
    assert!((mc.rope_theta - 500000.0).abs() < 1e-6);
}
