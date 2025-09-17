//! c05_nlp_transformers: NLP和Transformer基础功能

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Token {
    pub text: String,
    pub id: u32,
}

impl Token {
    pub fn new(text: String, id: u32) -> Self {
        Self { text, id }
    }
}

/// 简单的分词器
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    next_id: u32,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            next_id: 0,
        }
    }
    
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            id
        } else {
            let id = self.next_id;
            self.vocab.insert(token.to_string(), id);
            self.reverse_vocab.insert(id, token.to_string());
            self.next_id += 1;
            id
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|token| self.vocab.get(token).copied().unwrap_or(0)) // 0 for unknown
            .collect()
    }
    
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.reverse_vocab.get(&id))
            .cloned()
            .collect::<Vec<String>>()
            .join(" ")
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// 位置编码（简化版）
pub fn positional_encoding(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    let mut pe = vec![vec![0.0; d_model]; seq_len];
    
    for pos in 0..seq_len {
        for i in 0..d_model {
            if i % 2 == 0 {
                pe[pos][i] = (pos as f64 / 10000.0_f64.powf(i as f64 / d_model as f64)).sin();
            } else {
                pe[pos][i] = (pos as f64 / 10000.0_f64.powf((i - 1) as f64 / d_model as f64)).cos();
            }
        }
    }
    
    pe
}

/// 注意力机制（简化版）
pub fn scaled_dot_product_attention(
    query: &[Vec<f64>],
    key: &[Vec<f64>],
    value: &[Vec<f64>],
    mask: Option<&[Vec<f64>]>,
) -> Vec<Vec<f64>> {
    let seq_len = query.len();
    let d_k = query[0].len();
    let scale = (d_k as f64).sqrt();
    
    // 计算注意力分数
    let mut scores = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            scores[i][j] = dot_product(&query[i], &key[j]) / scale;
        }
    }
    
    // 应用mask（如果提供）
    if let Some(mask) = mask {
        for i in 0..seq_len {
            for j in 0..seq_len {
                scores[i][j] += mask[i][j];
            }
        }
    }
    
    // Softmax
    let attention_weights = softmax_2d(&scores);
    
    // 计算输出
    let mut output = vec![vec![0.0; value[0].len()]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            for k in 0..value[0].len() {
                output[i][k] += attention_weights[i][j] * value[j][k];
            }
        }
    }
    
    output
}

/// 点积
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 2D Softmax
fn softmax_2d(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    input.iter().map(|row| softmax_1d(row)).collect()
}

/// 1D Softmax
fn softmax_1d(input: &[f64]) -> Vec<f64> {
    let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f64 = input.iter().map(|&x| (x - max_val).exp()).sum();
    input.iter().map(|&x| (x - max_val).exp() / exp_sum).collect()
}

/// 多头注意力（简化版）
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, d_model: usize) -> Self {
        Self {
            num_heads,
            d_model,
            d_k: d_model / num_heads,
        }
    }
    
    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // 简化的多头注意力实现
        // 实际实现中需要线性变换和头分割
        scaled_dot_product_attention(query, key, value, None)
    }
}

/// 前馈网络
pub struct FeedForward {
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self { d_model, d_ff }
    }
    
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // 简化的前馈网络实现
        // 实际实现中需要线性变换和激活函数
        input.to_vec()
    }
}

/// Transformer编码器层（简化版）
pub struct TransformerEncoderLayer {
    pub attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub d_model: usize,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(num_heads, d_model),
            feed_forward: FeedForward::new(d_model, d_ff),
            d_model,
        }
    }
    
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // 自注意力
        let attention_output = self.attention.forward(input, input, input);
        
        // 残差连接和层归一化（简化）
        let mut output = vec![vec![0.0; self.d_model]; input.len()];
        for i in 0..input.len() {
            for j in 0..self.d_model {
                output[i][j] = input[i][j] + attention_output[i][j];
            }
        }
        
        // 前馈网络
        let ff_output = self.feed_forward.forward(&output);
        
        // 残差连接和层归一化（简化）
        for i in 0..input.len() {
            for j in 0..self.d_model {
                output[i][j] = output[i][j] + ff_output[i][j];
            }
        }
        
        output
    }
}

/// 文本嵌入（简化版）
pub fn text_embedding(text: &str, vocab_size: usize, d_model: usize) -> Vec<Vec<f64>> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut embeddings = vec![vec![0.0; d_model]; tokens.len()];
    
    // 简单的词嵌入（实际中应该使用预训练的嵌入）
    for (i, token) in tokens.iter().enumerate() {
        let hash = token.len() as u32;
        for j in 0..d_model {
            embeddings[i][j] = ((hash + j as u32) as f64 / vocab_size as f64) * 2.0 - 1.0;
        }
    }
    
    embeddings
}

pub fn whitespace_tokenize(text: &str) -> Vec<Token> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, s)| Token::new(s.to_string(), i as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_basic() {
        let toks = whitespace_tokenize("hello world");
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].text, "hello");
    }
    
    #[test]
    fn test_tokenizer() {
        let mut tokenizer = SimpleTokenizer::new();
        
        // 添加词汇
        tokenizer.add_token("hello");
        tokenizer.add_token("world");
        tokenizer.add_token("rust");
        
        // 编码
        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded.len(), 2);
        
        // 解码
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello world");
        
        assert_eq!(tokenizer.vocab_size(), 3);
    }
    
    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding(3, 4);
        assert_eq!(pe.len(), 3);
        assert_eq!(pe[0].len(), 4);
        
        // 检查位置编码的周期性
        assert!(pe[0][0] != pe[1][0]);
    }
    
    #[test]
    fn test_attention() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let key = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let value = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        
        let output = scaled_dot_product_attention(&query, &key, &value, None);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 2);
    }
    
    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax_1d(&input);
        
        // 检查softmax属性
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(output.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_multi_head_attention() {
        let attention = MultiHeadAttention::new(2, 4);
        assert_eq!(attention.num_heads, 2);
        assert_eq!(attention.d_model, 4);
        assert_eq!(attention.d_k, 2);
        
        let input = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let output = attention.forward(&input, &input, &input);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
    }
    
    #[test]
    fn test_transformer_encoder_layer() {
        let layer = TransformerEncoderLayer::new(4, 2, 8);
        let input = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let output = layer.forward(&input);
        
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
    }
    
    #[test]
    fn test_text_embedding() {
        let embeddings = text_embedding("hello world", 100, 4);
        assert_eq!(embeddings.len(), 2); // 两个词
        assert_eq!(embeddings[0].len(), 4); // 4维嵌入
        
        // 检查嵌入值在合理范围内
        for embedding in &embeddings {
            for &value in embedding {
                assert!(value >= -1.0 && value <= 1.0);
            }
        }
    }
}


