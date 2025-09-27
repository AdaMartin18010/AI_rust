//! 内存优化模块
//! 
//! 提供零拷贝、内存池、缓存优化等内存管理功能

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// 内存池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    pub initial_size: usize,
    pub max_size: usize,
    pub block_size: usize,
    pub growth_factor: f64,
    pub cleanup_threshold: usize,
    pub cleanup_interval: Duration,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024,      // 1MB
            max_size: 1024 * 1024 * 100,    // 100MB
            block_size: 4096,               // 4KB
            growth_factor: 1.5,
            cleanup_threshold: 1000,
            cleanup_interval: Duration::from_secs(300), // 5分钟
        }
    }
}

/// 内存块
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub data: Vec<u8>,
    pub size: usize,
    pub allocated: bool,
    pub last_used: Instant,
    pub usage_count: u64,
}

impl MemoryBlock {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
            size,
            allocated: false,
            last_used: Instant::now(),
            usage_count: 0,
        }
    }
    
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_used.elapsed() > timeout
    }
    
    pub fn reset(&mut self) {
        self.allocated = false;
        self.last_used = Instant::now();
        self.usage_count += 1;
    }
}

/// 内存池
pub struct MemoryPool {
    config: MemoryPoolConfig,
    blocks: Arc<RwLock<HashMap<usize, Vec<MemoryBlock>>>>,
    stats: Arc<Mutex<MemoryPoolStats>>,
}

/// 内存池统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub total_memory: usize,
    pub used_memory: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl Default for MemoryPoolStats {
    fn default() -> Self {
        Self {
            total_blocks: 0,
            allocated_blocks: 0,
            free_blocks: 0,
            total_memory: 0,
            used_memory: 0,
            allocation_count: 0,
            deallocation_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl MemoryPool {
    /// 创建新的内存池
    pub fn new(config: MemoryPoolConfig) -> Self {
        let pool = Self {
            config,
            blocks: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryPoolStats::default())),
        };
        
        // 初始化内存池
        pool.initialize_pool();
        pool
    }
    
    /// 分配内存块
    pub fn allocate(&self, size: usize) -> Option<Vec<u8>> {
        let mut blocks = self.blocks.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        // 查找合适大小的内存块
        if let Some(block_list) = blocks.get_mut(&size) {
            if let Some(block) = block_list.iter_mut().find(|b| !b.allocated) {
                block.allocated = true;
                block.last_used = Instant::now();
                stats.allocated_blocks += 1;
                stats.used_memory += size;
                stats.allocation_count += 1;
                stats.cache_hits += 1;
                
                return Some(block.data.clone());
            }
        }
        
        // 如果没有找到合适的块，创建新的
        let new_block = MemoryBlock::new(size);
        blocks.entry(size).or_insert_with(Vec::new).push(new_block.clone());
        
        stats.total_blocks += 1;
        stats.allocated_blocks += 1;
        stats.total_memory += size;
        stats.used_memory += size;
        stats.allocation_count += 1;
        stats.cache_misses += 1;
        
        Some(new_block.data)
    }
    
    /// 释放内存块
    pub fn deallocate(&self, _data: Vec<u8>, size: usize) {
        let mut blocks = self.blocks.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        if let Some(block_list) = blocks.get_mut(&size) {
            if let Some(block) = block_list.iter_mut().find(|b| b.allocated) {
                block.allocated = false;
                block.last_used = Instant::now();
                stats.allocated_blocks -= 1;
                stats.used_memory -= size;
                stats.deallocation_count += 1;
            }
        }
    }
    
    /// 获取内存池统计信息
    pub fn get_stats(&self) -> MemoryPoolStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// 清理过期内存块
    pub fn cleanup(&self) {
        let mut blocks = self.blocks.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        for (size, block_list) in blocks.iter_mut() {
            let original_len = block_list.len();
            block_list.retain(|block| {
                if !block.allocated && block.is_expired(Duration::from_secs(300)) {
                    false
                } else {
                    true
                }
            });
            
            let cleaned = original_len - block_list.len();
            if cleaned > 0 {
                stats.total_blocks -= cleaned;
                stats.total_memory -= cleaned * size;
            }
        }
        
        stats.free_blocks = stats.total_blocks - stats.allocated_blocks;
    }
    
    /// 初始化内存池
    fn initialize_pool(&self) {
        let mut blocks = self.blocks.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        // 创建初始内存块
        let initial_blocks = self.config.initial_size / self.config.block_size;
        
        for _ in 0..initial_blocks {
            let block = MemoryBlock::new(self.config.block_size);
            blocks.entry(self.config.block_size).or_insert_with(Vec::new).push(block);
        }
        
        stats.total_blocks = initial_blocks;
        stats.free_blocks = initial_blocks;
        stats.total_memory = self.config.initial_size;
    }
}

/// 零拷贝缓冲区
#[derive(Clone)]
pub struct ZeroCopyBuffer {
    data: Arc<[u8]>,
    offset: usize,
    length: usize,
}

impl ZeroCopyBuffer {
    /// 创建新的零拷贝缓冲区
    pub fn new(data: Vec<u8>) -> Self {
        let length = data.len();
        Self {
            data: data.into_boxed_slice().into(),
            offset: 0,
            length,
        }
    }
    
    /// 创建子缓冲区
    pub fn slice(&self, start: usize, end: usize) -> Option<Self> {
        if start < end && end <= self.length {
            Some(Self {
                data: self.data.clone(),
                offset: self.offset + start,
                length: end - start,
            })
        } else {
            None
        }
    }
    
    /// 获取数据引用
    pub fn as_slice(&self) -> &[u8] {
        &self.data[self.offset..self.offset + self.length]
    }
    
    /// 获取长度
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// 缓存管理器
pub struct CacheManager {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_size: usize,
    current_size: Arc<Mutex<usize>>,
    stats: Arc<Mutex<CacheStats>>,
}

/// 缓存条目
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    size: usize,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
}

/// 缓存统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub hit_rate: f64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            total_size: 0,
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
            hit_rate: 0.0,
        }
    }
}

impl CacheManager {
    /// 创建新的缓存管理器
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            current_size: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }
    
    /// 获取缓存数据
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        if let Some(entry) = cache.get_mut(key) {
            // 检查是否过期
            if let Some(ttl) = entry.ttl {
                if entry.created_at.elapsed() > ttl {
                    cache.remove(key);
                    stats.eviction_count += 1;
                    stats.miss_count += 1;
                    return None;
                }
            }
            
            // 更新访问信息
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            stats.hit_count += 1;
            
            // 更新命中率
            stats.hit_rate = stats.hit_count as f64 / (stats.hit_count + stats.miss_count) as f64;
            
            Some(entry.data.clone())
        } else {
            stats.miss_count += 1;
            stats.hit_rate = stats.hit_count as f64 / (stats.hit_count + stats.miss_count) as f64;
            None
        }
    }
    
    /// 设置缓存数据
    pub fn set(&self, key: String, data: Vec<u8>, ttl: Option<Duration>) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        let size = data.len();
        let mut current_size = *self.current_size.lock().unwrap();
        
        // 检查是否需要清理空间
        while current_size + size > self.max_size && !cache.is_empty() {
            self.evict_lru(&mut cache, &mut stats);
            current_size = *self.current_size.lock().unwrap();
        }
        
        let entry = CacheEntry {
            data,
            size,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            ttl,
        };
        
        // 如果键已存在，先移除旧条目
        if let Some(old_entry) = cache.remove(&key) {
            *self.current_size.lock().unwrap() -= old_entry.size;
        }
        
        cache.insert(key, entry);
        *self.current_size.lock().unwrap() += size;
        
        stats.total_entries = cache.len();
        stats.total_size = *self.current_size.lock().unwrap();
    }
    
    /// 删除缓存数据
    pub fn remove(&self, key: &str) -> bool {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        if let Some(entry) = cache.remove(key) {
            *self.current_size.lock().unwrap() -= entry.size;
            stats.total_entries = cache.len();
            stats.total_size = *self.current_size.lock().unwrap();
            true
        } else {
            false
        }
    }
    
    /// 清理过期缓存
    pub fn cleanup_expired(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        let mut removed_keys = Vec::new();
        
        for (key, entry) in cache.iter() {
            if let Some(ttl) = entry.ttl {
                if entry.created_at.elapsed() > ttl {
                    removed_keys.push(key.clone());
                }
            }
        }
        
        for key in removed_keys {
            if let Some(entry) = cache.remove(&key) {
                *self.current_size.lock().unwrap() -= entry.size;
                stats.eviction_count += 1;
            }
        }
        
        stats.total_entries = cache.len();
        stats.total_size = *self.current_size.lock().unwrap();
    }
    
    /// 获取缓存统计信息
    pub fn get_stats(&self) -> CacheStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }
    
    /// 清除所有缓存
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.lock().unwrap();
        
        cache.clear();
        *self.current_size.lock().unwrap() = 0;
        
        stats.total_entries = 0;
        stats.total_size = 0;
    }
    
    // 私有方法
    
    /// 驱逐最近最少使用的条目
    fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>, stats: &mut CacheStats) {
        if let Some((lru_key, _)) = cache.iter()
            .min_by_key(|(_, entry)| entry.last_accessed) {
            let key = lru_key.clone();
            if let Some(entry) = cache.remove(&key) {
                *self.current_size.lock().unwrap() -= entry.size;
                stats.eviction_count += 1;
            }
        }
    }
}

/// 内存优化管理器
pub struct MemoryOptimizer {
    memory_pool: Arc<MemoryPool>,
    cache_manager: Arc<CacheManager>,
    zero_copy_buffers: Arc<RwLock<HashMap<String, ZeroCopyBuffer>>>,
    stats: Arc<Mutex<MemoryOptimizerStats>>,
}

/// 内存优化统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizerStats {
    pub pool_stats: MemoryPoolStats,
    pub cache_stats: CacheStats,
    pub zero_copy_count: usize,
    pub total_memory_saved: usize,
    pub optimization_ratio: f64,
}

impl Default for MemoryOptimizerStats {
    fn default() -> Self {
        Self {
            pool_stats: MemoryPoolStats::default(),
            cache_stats: CacheStats::default(),
            zero_copy_count: 0,
            total_memory_saved: 0,
            optimization_ratio: 0.0,
        }
    }
}

impl MemoryOptimizer {
    /// 创建新的内存优化管理器
    pub fn new(pool_config: MemoryPoolConfig, cache_size: usize) -> Self {
        Self {
            memory_pool: Arc::new(MemoryPool::new(pool_config)),
            cache_manager: Arc::new(CacheManager::new(cache_size)),
            zero_copy_buffers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryOptimizerStats::default())),
        }
    }
    
    /// 获取内存池
    pub fn get_memory_pool(&self) -> Arc<MemoryPool> {
        self.memory_pool.clone()
    }
    
    /// 获取缓存管理器
    pub fn get_cache_manager(&self) -> Arc<CacheManager> {
        self.cache_manager.clone()
    }
    
    /// 创建零拷贝缓冲区
    pub fn create_zero_copy_buffer(&self, key: String, data: Vec<u8>) {
        let buffer = ZeroCopyBuffer::new(data);
        let mut buffers = self.zero_copy_buffers.write().unwrap();
        buffers.insert(key, buffer);
        
        let mut stats = self.stats.lock().unwrap();
        stats.zero_copy_count = buffers.len();
    }
    
    /// 获取零拷贝缓冲区
    pub fn get_zero_copy_buffer(&self, key: &str) -> Option<ZeroCopyBuffer> {
        let buffers = self.zero_copy_buffers.read().unwrap();
        buffers.get(key).cloned()
    }
    
    /// 优化内存使用
    pub fn optimize(&self) {
        // 清理内存池
        self.memory_pool.cleanup();
        
        // 清理过期缓存
        self.cache_manager.cleanup_expired();
        
        // 更新统计信息
        let mut stats = self.stats.lock().unwrap();
        stats.pool_stats = self.memory_pool.get_stats();
        stats.cache_stats = self.cache_manager.get_stats();
        stats.zero_copy_count = self.zero_copy_buffers.read().unwrap().len();
        
        // 计算优化比例
        let total_memory = stats.pool_stats.total_memory + stats.cache_stats.total_size;
        stats.total_memory_saved = total_memory.saturating_sub(stats.pool_stats.used_memory + stats.cache_stats.total_size);
        stats.optimization_ratio = if total_memory > 0 {
            stats.total_memory_saved as f64 / total_memory as f64
        } else {
            0.0
        };
    }
    
    /// 获取内存优化统计信息
    pub fn get_stats(&self) -> MemoryOptimizerStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }
    
    /// 生成内存优化报告
    pub fn generate_report(&self) -> String {
        let stats = self.get_stats();
        
        format!(
            "# 内存优化报告\n\n\
            ## 内存池统计\n\
            - 总块数: {}\n\
            - 已分配块数: {}\n\
            - 空闲块数: {}\n\
            - 总内存: {} bytes\n\
            - 已用内存: {} bytes\n\
            - 分配次数: {}\n\
            - 释放次数: {}\n\
            - 缓存命中: {}\n\
            - 缓存未命中: {}\n\n\
            ## 缓存统计\n\
            - 总条目数: {}\n\
            - 总大小: {} bytes\n\
            - 命中次数: {}\n\
            - 未命中次数: {}\n\
            - 驱逐次数: {}\n\
            - 命中率: {:.2}%\n\n\
            ## 零拷贝统计\n\
            - 零拷贝缓冲区数: {}\n\
            - 节省内存: {} bytes\n\
            - 优化比例: {:.2}%\n",
            stats.pool_stats.total_blocks,
            stats.pool_stats.allocated_blocks,
            stats.pool_stats.free_blocks,
            stats.pool_stats.total_memory,
            stats.pool_stats.used_memory,
            stats.pool_stats.allocation_count,
            stats.pool_stats.deallocation_count,
            stats.pool_stats.cache_hits,
            stats.pool_stats.cache_misses,
            stats.cache_stats.total_entries,
            stats.cache_stats.total_size,
            stats.cache_stats.hit_count,
            stats.cache_stats.miss_count,
            stats.cache_stats.eviction_count,
            stats.cache_stats.hit_rate * 100.0,
            stats.zero_copy_count,
            stats.total_memory_saved,
            stats.optimization_ratio * 100.0
        )
    }
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new(MemoryPoolConfig::default(), 100 * 1024 * 1024) // 100MB缓存
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);
        let stats = pool.get_stats();
        
        assert!(stats.total_blocks > 0);
        assert_eq!(stats.allocated_blocks, 0);
    }
    
    #[test]
    fn test_memory_pool_allocation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);
        
        let data = pool.allocate(4096);
        assert!(data.is_some());
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocated_blocks, 1);
        assert_eq!(stats.used_memory, 4096);
    }
    
    #[test]
    fn test_zero_copy_buffer() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let buffer = ZeroCopyBuffer::new(data);
        
        assert_eq!(buffer.len(), 10);
        assert!(!buffer.is_empty());
        
        let slice = buffer.slice(2, 8);
        assert!(slice.is_some());
        
        let slice = slice.unwrap();
        assert_eq!(slice.len(), 6);
        assert_eq!(slice.as_slice(), &[3, 4, 5, 6, 7, 8]);
    }
    
    #[test]
    fn test_cache_manager() {
        let cache = CacheManager::new(1024);
        
        cache.set("key1".to_string(), vec![1, 2, 3], None);
        let data = cache.get("key1");
        assert!(data.is_some());
        assert_eq!(data.unwrap(), vec![1, 2, 3]);
        
        let stats = cache.get_stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.hit_count, 1);
    }
    
    #[test]
    fn test_memory_optimizer() {
        let optimizer = MemoryOptimizer::default();
        
        optimizer.create_zero_copy_buffer("test".to_string(), vec![1, 2, 3, 4, 5]);
        
        let buffer = optimizer.get_zero_copy_buffer("test");
        assert!(buffer.is_some());
        assert_eq!(buffer.unwrap().len(), 5);
        
        optimizer.optimize();
        let stats = optimizer.get_stats();
        assert_eq!(stats.zero_copy_count, 1);
    }
}