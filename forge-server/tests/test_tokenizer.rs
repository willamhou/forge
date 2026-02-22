use forge_server::chat_template::ChatTemplate;

#[test]
fn test_template_with_eos_bos_tokens() {
    let tpl = "{{ bos_token }}{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}{{ eos_token }}";
    let template = ChatTemplate::with_tokens(tpl, "<s>", "</s>").unwrap();
    let messages = vec![("user", "hello")];
    let rendered = template.apply(&messages, false).unwrap();
    assert_eq!(rendered, "<s>user: hello</s>");
}

#[test]
fn test_template_with_empty_tokens() {
    let tpl = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}";
    let template = ChatTemplate::new(tpl).unwrap();
    let messages = vec![("user", "hi")];
    let rendered = template.apply(&messages, false).unwrap();
    assert_eq!(rendered, "hi");
}

#[test]
fn test_chatml_template() {
    let template = ChatTemplate::chatml_default().unwrap();
    let messages = vec![("system", "You are helpful."), ("user", "Hello!")];
    let rendered = template.apply(&messages, true).unwrap();
    assert!(rendered.contains("<|im_start|>system"));
    assert!(rendered.contains("You are helpful."));
    assert!(rendered.contains("<|im_start|>user"));
    assert!(rendered.contains("Hello!"));
    assert!(rendered.contains("<|im_start|>assistant"));
}

#[test]
fn test_chatml_no_generation_prompt() {
    let template = ChatTemplate::chatml_default().unwrap();
    let messages = vec![("user", "Hi")];
    let rendered = template.apply(&messages, false).unwrap();
    assert!(rendered.contains("<|im_start|>user"));
    assert!(!rendered.contains("<|im_start|>assistant"));
}

#[test]
fn test_custom_template() {
    let tpl = "{% for m in messages %}[{{ m.role }}]: {{ m.content }}\n{% endfor %}";
    let template = ChatTemplate::new(tpl).unwrap();
    let messages = vec![("user", "test")];
    let rendered = template.apply(&messages, false).unwrap();
    assert_eq!(rendered.trim(), "[user]: test");
}
