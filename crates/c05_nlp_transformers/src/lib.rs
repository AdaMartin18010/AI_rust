//! c05_nlp_transformers: 简化分词占位与推理接口草图

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Token(pub String);

pub fn whitespace_tokenize(text: &str) -> Vec<Token> {
    text.split_whitespace().map(|s| Token(s.to_string())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_basic() {
        let toks = whitespace_tokenize("hello world");
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].0, "hello");
    }
}


