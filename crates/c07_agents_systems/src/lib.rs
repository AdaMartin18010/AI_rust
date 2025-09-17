//! c07_agents_systems: 简易任务计划与记忆占位

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Task {
    pub id: u64,
    pub desc: String,
}

pub fn simple_plan(prompt: &str) -> Vec<Task> {
    let steps = ["理解需求", "检索资料", "执行动作", "验证与总结"];
    steps
        .iter()
        .enumerate()
        .map(|(i, s)| Task { id: i as u64 + 1, desc: format!("{}: {}", s, prompt) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_has_steps() {
        let p = simple_plan("build rag");
        assert_eq!(p.len(), 4);
    }
}


