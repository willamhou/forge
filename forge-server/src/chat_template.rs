use forge_core::{ForgeError, Result};
use minijinja::Environment;

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(template_str: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.add_template_owned("chat", template_str.to_string())
            .map_err(|e| ForgeError::Internal(format!("Template parse error: {e}")))?;
        Ok(Self { env })
    }

    pub fn chatml_default() -> Result<Self> {
        Self::new(CHATML_TEMPLATE)
    }

    pub fn apply(&self, messages: &[(&str, &str)], add_generation_prompt: bool) -> Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| ForgeError::Internal(e.to_string()))?;

        let msgs: Vec<minijinja::Value> = messages
            .iter()
            .map(|(role, content)| minijinja::context! { role => *role, content => *content })
            .collect();

        tmpl.render(minijinja::context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
        })
        .map_err(|e| ForgeError::Internal(e.to_string()))
    }
}

const CHATML_TEMPLATE: &str = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;
