//! c04_dl_fundamentals: 张量与自动微分的占位（后续可接 candle）

pub fn relu(x: f64) -> f64 { if x > 0.0 { x } else { 0.0 } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relu_works() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(3.5), 3.5);
    }
}


