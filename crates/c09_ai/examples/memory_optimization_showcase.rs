//! 内存优化展示 - 零拷贝、内存池、缓存优化
//! 
//! 本示例展示了AI-Rust项目的内存优化功能，包括：
//! - 内存池管理
//! - 零拷贝缓冲区
//! - 智能缓存系统
//! - 内存使用优化

use c19_ai::*;
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志系统
    tracing_subscriber::fmt::init();
    
    println!("🧠 AI-Rust 内存优化展示");
    println!("🎯 目标：展示零拷贝、内存池、缓存优化功能");
    println!("{}", "=".repeat(60));

    // 创建AI引擎
    let mut engine = create_memory_optimized_engine().await?;
    
    // 展示内存池管理
    demonstrate_memory_pool_management(&mut engine).await?;
    
    // 展示零拷贝缓冲区
    demonstrate_zero_copy_buffers(&mut engine).await?;
    
    // 展示智能缓存系统
    demonstrate_smart_cache_system(&mut engine).await?;
    
    // 展示内存使用优化
    demonstrate_memory_usage_optimization(&mut engine).await?;
    
    // 展示内存性能分析
    demonstrate_memory_performance_analysis(&mut engine).await?;
    
    println!("\n🎉 内存优化展示完成！");
    println!("📊 内存优化总结：");
    print_memory_optimization_summary(&engine).await;
    
    // 清理资源
    engine.cleanup()?;
    println!("✅ 资源清理完成");
    
    Ok(())
}

/// 创建内存优化引擎
async fn create_memory_optimized_engine() -> Result<AIEngine, Error> {
    println!("\n🔧 创建内存优化引擎...");
    
    let mut config = EngineConfig::default();
    config.enable_gpu = true;
    config.enable_monitoring = true;
    config.max_models = 15;
    config.cache_size = 50000; // 大缓存
    
    let mut engine = AIEngine::with_config(config);
    
    // 设置内存优化相关状态
    engine.set_state("memory_optimization", "enabled")?;
    engine.set_state("zero_copy_mode", "enabled")?;
    engine.set_state("cache_strategy", "lru")?;
    engine.set_state("memory_pool_size", "100MB")?;
    
    println!("✅ 内存优化引擎创建完成");
    Ok(engine)
}

/// 展示内存池管理
async fn demonstrate_memory_pool_management(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n🏊 展示内存池管理...");
    
    let memory_pool = engine.get_memory_pool();
    
    // 测试不同大小的内存分配
    let allocation_sizes = vec![1024, 4096, 16384, 65536, 262144]; // 1KB到256KB
    
    for size in allocation_sizes {
        println!("🔄 分配内存块: {} bytes", size);
        
        let start_time = Instant::now();
        
        // 分配内存
        if let Some(data) = memory_pool.allocate(size) {
            let allocation_time = start_time.elapsed();
            
            println!("   ✅ 分配成功，耗时: {:.2}μs", allocation_time.as_micros());
            println!("   📊 数据长度: {} bytes", data.len());
            
            // 模拟使用内存
            for i in 0..data.len().min(10) {
                let _ = data[i]; // 模拟访问
            }
            
            // 释放内存
            let deallocation_start = Instant::now();
            memory_pool.deallocate(data, size);
            let deallocation_time = deallocation_start.elapsed();
            
            println!("   🗑️  释放完成，耗时: {:.2}μs", deallocation_time.as_micros());
        } else {
            println!("   ❌ 分配失败");
        }
        
        sleep(Duration::from_millis(50)).await;
    }
    
    // 显示内存池统计信息
    let pool_stats = memory_pool.get_stats();
    println!("📊 内存池统计信息:");
    println!("   • 总块数: {}", pool_stats.total_blocks);
    println!("   • 已分配块数: {}", pool_stats.allocated_blocks);
    println!("   • 空闲块数: {}", pool_stats.free_blocks);
    println!("   • 总内存: {} bytes ({:.2} MB)", 
            pool_stats.total_memory, pool_stats.total_memory as f64 / 1024.0 / 1024.0);
    println!("   • 已用内存: {} bytes ({:.2} MB)", 
            pool_stats.used_memory, pool_stats.used_memory as f64 / 1024.0 / 1024.0);
    println!("   • 分配次数: {}", pool_stats.allocation_count);
    println!("   • 缓存命中率: {:.2}%", 
            (pool_stats.cache_hits as f64 / (pool_stats.cache_hits + pool_stats.cache_misses) as f64) * 100.0);
    
    // 清理内存池
    memory_pool.cleanup();
    println!("🧹 内存池清理完成");
    
    Ok(())
}

/// 展示零拷贝缓冲区
async fn demonstrate_zero_copy_buffers(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n🚀 展示零拷贝缓冲区...");
    
    // 创建大量数据
    let large_data = vec![42u8; 1024 * 1024]; // 1MB数据
    println!("📦 创建大数据块: {} bytes", large_data.len());
    
    // 创建零拷贝缓冲区
    let buffer_key = "large_data_buffer".to_string();
    engine.create_zero_copy_buffer(buffer_key.clone(), large_data);
    println!("✅ 零拷贝缓冲区创建完成");
    
    // 获取零拷贝缓冲区
    if let Some(buffer) = engine.get_zero_copy_buffer(&buffer_key) {
        println!("📊 缓冲区信息:");
        println!("   • 总长度: {} bytes", buffer.len());
        println!("   • 是否为空: {}", buffer.is_empty());
        
        // 创建子缓冲区（零拷贝切片）
        let slice_sizes = vec![1024, 4096, 16384, 65536];
        
        for slice_size in slice_sizes {
            if let Some(slice) = buffer.slice(0, slice_size) {
                let start_time = Instant::now();
                
                // 模拟处理切片数据
                let data_slice = slice.as_slice();
                let _sum: u64 = data_slice.iter().map(|&x| x as u64).sum();
                
                let processing_time = start_time.elapsed();
                
                println!("   🔪 切片 {} bytes: 处理耗时 {:.2}μs, 求和结果: {}", 
                        slice_size, processing_time.as_micros(), _sum);
            }
        }
        
        // 创建多个重叠切片
        println!("   🔄 创建重叠切片:");
        for i in 0..5 {
            let start = i * 10000;
            let end = start + 5000;
            if let Some(slice) = buffer.slice(start, end) {
                println!("      • 切片 {}: {} bytes (位置 {} - {})", 
                        i + 1, slice.len(), start, end);
            }
        }
    }
    
    // 创建多个零拷贝缓冲区
    println!("\n📚 创建多个零拷贝缓冲区:");
    let buffer_count = 10;
    
    for i in 0..buffer_count {
        let key = format!("buffer_{}", i);
        let data = vec![i as u8; 1024 * 100]; // 100KB数据
        engine.create_zero_copy_buffer(key.clone(), data);
        
        if let Some(buffer) = engine.get_zero_copy_buffer(&key) {
            println!("   • {}: {} bytes", key, buffer.len());
        }
    }
    
    // 显示零拷贝统计
    let memory_stats = engine.get_memory_stats();
    println!("📈 零拷贝统计:");
    println!("   • 缓冲区数量: {}", memory_stats.zero_copy_count);
    
    Ok(())
}

/// 展示智能缓存系统
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_smart_cache_system(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n🧠 展示智能缓存系统...");
    
    let cache_manager = engine.get_cache_manager();
    
    // 测试缓存基本操作
    println!("🔄 测试缓存基本操作:");
    
    let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let cache_key = "test_data".to_string();
    
    // 设置缓存
    cache_manager.set(cache_key.clone(), test_data.clone(), Some(Duration::from_secs(60)));
    println!("   ✅ 数据已缓存: {} bytes", test_data.len());
    
    // 获取缓存
    if let Some(cached_data) = cache_manager.get(&cache_key) {
        println!("   📥 缓存命中: {} bytes", cached_data.len());
        assert_eq!(cached_data, test_data);
    } else {
        println!("   ❌ 缓存未命中");
    }
    
    // 测试缓存过期
    println!("🕐 测试缓存过期:");
    let short_ttl_key = "short_ttl_data".to_string();
    let short_data = vec![42; 1000];
    
    cache_manager.set(short_ttl_key.clone(), short_data, Some(Duration::from_millis(100)));
    println!("   ⏰ 设置短期缓存 (100ms TTL)");
    
    sleep(Duration::from_millis(50)).await;
    if cache_manager.get(&short_ttl_key).is_some() {
        println!("   ✅ 缓存仍然有效");
    }
    
    sleep(Duration::from_millis(100)).await;
    if cache_manager.get(&short_ttl_key).is_some() {
        println!("   ❌ 缓存应该已过期但仍存在");
    } else {
        println!("   ✅ 缓存已正确过期");
    }
    
    // 测试缓存容量限制
    println!("📊 测试缓存容量限制:");
    let cache_size = 1024 * 1024; // 1MB
    let large_data_size = 1024 * 500; // 500KB per item
    
    for i in 0..5 {
        let key = format!("large_item_{}", i);
        let data = vec![i as u8; large_data_size];
        cache_manager.set(key.clone(), data, None);
        println!("   📦 缓存大项目 {}: {} bytes", i + 1, large_data_size);
        
        let stats = cache_manager.get_stats();
        println!("      • 当前缓存大小: {} bytes ({:.2} MB)", 
                stats.total_size, stats.total_size as f64 / 1024.0 / 1024.0);
    }
    
    // 清理过期缓存
    cache_manager.cleanup_expired();
    println!("🧹 清理过期缓存完成");
    
    // 显示缓存统计信息
    let final_stats = cache_manager.get_stats();
    println!("📈 缓存统计信息:");
    println!("   • 总条目数: {}", final_stats.total_entries);
    println!("   • 总大小: {} bytes ({:.2} MB)", 
            final_stats.total_size, final_stats.total_size as f64 / 1024.0 / 1024.0);
    println!("   • 命中次数: {}", final_stats.hit_count);
    println!("   • 未命中次数: {}", final_stats.miss_count);
    println!("   • 驱逐次数: {}", final_stats.eviction_count);
    println!("   • 命中率: {:.2}%", final_stats.hit_rate * 100.0);
    
    Ok(())
}

/// 展示内存使用优化
async fn demonstrate_memory_usage_optimization(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n⚡ 展示内存使用优化...");
    
    // 获取优化前的内存统计
    let stats_before = engine.get_memory_stats();
    println!("📊 优化前内存状态:");
    println!("   • 内存池使用: {} bytes", stats_before.pool_stats.used_memory);
    println!("   • 缓存使用: {} bytes", stats_before.cache_stats.total_size);
    println!("   • 零拷贝缓冲区: {}", stats_before.zero_copy_count);
    
    // 执行内存优化
    println!("🔧 执行内存优化...");
    let optimization_start = Instant::now();
    engine.optimize_memory();
    let optimization_time = optimization_start.elapsed();
    
    println!("   ✅ 优化完成，耗时: {:.2}ms", optimization_time.as_millis());
    
    // 获取优化后的内存统计
    let stats_after = engine.get_memory_stats();
    println!("📊 优化后内存状态:");
    println!("   • 内存池使用: {} bytes", stats_after.pool_stats.used_memory);
    println!("   • 缓存使用: {} bytes", stats_after.cache_stats.total_size);
    println!("   • 零拷贝缓冲区: {}", stats_after.zero_copy_count);
    println!("   • 节省内存: {} bytes", stats_after.total_memory_saved);
    println!("   • 优化比例: {:.2}%", stats_after.optimization_ratio * 100.0);
    
    // 模拟内存压力测试
    println!("💪 内存压力测试:");
    let stress_test_iterations = 100;
    
    for i in 0..stress_test_iterations {
        // 创建临时数据
        let temp_data = vec![i as u8; 1024];
        let temp_key = format!("stress_test_{}", i);
        
        // 使用缓存
        engine.get_cache_manager().set(temp_key.clone(), temp_data, Some(Duration::from_secs(1)));
        
        // 使用内存池
        if let Some(_pooled_data) = engine.get_memory_pool().allocate(1024) {
            // 模拟使用
        }
        
        // 创建零拷贝缓冲区
        engine.create_zero_copy_buffer(format!("zero_copy_{}", i), vec![i as u8; 512]);
        
        if i % 20 == 0 {
            // 定期优化
            engine.optimize_memory();
            let current_stats = engine.get_memory_stats();
            println!("   📈 第 {} 轮: 总内存 {} bytes, 优化比例 {:.2}%", 
                    i + 1, 
                    current_stats.pool_stats.total_memory + current_stats.cache_stats.total_size,
                    current_stats.optimization_ratio * 100.0);
        }
    }
    
    // 最终优化
    engine.optimize_memory();
    let final_stats = engine.get_memory_stats();
    
    println!("🎯 压力测试完成:");
    println!("   • 最终内存使用: {} bytes", 
            final_stats.pool_stats.used_memory + final_stats.cache_stats.total_size);
    println!("   • 最终优化比例: {:.2}%", final_stats.optimization_ratio * 100.0);
    
    Ok(())
}

/// 展示内存性能分析
#[allow(dead_code)]
#[allow(unused_variables)]
async fn demonstrate_memory_performance_analysis(engine: &mut AIEngine) -> Result<(), Error> {
    println!("\n📈 展示内存性能分析...");
    
    // 生成内存优化报告
    let report = engine.generate_memory_report();
    
    println!("📋 内存优化报告:");
    println!("{}", "=".repeat(60));
    
    // 显示报告的关键部分
    let lines: Vec<&str> = report.lines().take(30).collect();
    for line in lines {
        println!("{}", line);
    }
    
    if report.lines().count() > 30 {
        println!("... (报告包含更多详细信息)");
    }
    
    println!("{}", "=".repeat(60));
    
    // 性能对比测试
    println!("🏁 性能对比测试:");
    
    // 测试1: 传统内存分配 vs 内存池
    println!("🔄 测试1: 传统分配 vs 内存池分配");
    
    let iterations = 1000;
    let allocation_size = 4096;
    
    // 传统分配
    let traditional_start = Instant::now();
    let mut traditional_allocations = Vec::new();
    for _ in 0..iterations {
        traditional_allocations.push(vec![0u8; allocation_size]);
    }
    let traditional_time = traditional_start.elapsed();
    
    // 内存池分配
    let pool_start = Instant::now();
    let memory_pool = engine.get_memory_pool();
    let mut pool_allocations = Vec::new();
    for _ in 0..iterations {
        if let Some(data) = memory_pool.allocate(allocation_size) {
            pool_allocations.push(data);
        }
    }
    let pool_time = pool_start.elapsed();
    
    println!("   • 传统分配: {:.2}ms ({:.2}μs/次)", 
            traditional_time.as_millis(), 
            traditional_time.as_micros() as f64 / iterations as f64);
    println!("   • 内存池分配: {:.2}ms ({:.2}μs/次)", 
            pool_time.as_millis(), 
            pool_time.as_micros() as f64 / iterations as f64);
    
    let speedup = traditional_time.as_micros() as f64 / pool_time.as_micros() as f64;
    println!("   • 性能提升: {:.2}x", speedup);
    
    // 清理测试数据
    drop(traditional_allocations);
    drop(pool_allocations);
    
    // 测试2: 数据复制 vs 零拷贝
    println!("🔄 测试2: 数据复制 vs 零拷贝");
    
    let large_data = vec![42u8; 1024 * 1024]; // 1MB
    
    // 数据复制
    let copy_start = Instant::now();
    let _copied_data = large_data.clone();
    let copy_time = copy_start.elapsed();
    
    // 零拷贝
    let zero_copy_start = Instant::now();
    engine.create_zero_copy_buffer("performance_test".to_string(), large_data.clone());
    let zero_copy_buffer = engine.get_zero_copy_buffer("performance_test").unwrap();
    let zero_copy_time = zero_copy_start.elapsed();
    
    println!("   • 数据复制: {:.2}μs", copy_time.as_micros());
    println!("   • 零拷贝: {:.2}μs", zero_copy_time.as_micros());
    
    let zero_copy_speedup = copy_time.as_micros() as f64 / zero_copy_time.as_micros() as f64;
    println!("   • 零拷贝优势: {:.2}x", zero_copy_speedup);
    
    // 测试3: 缓存命中率测试
    println!("🔄 测试3: 缓存命中率测试");
    
    let cache_manager = engine.get_cache_manager();
    
    // 预热缓存
    for i in 0..10 {
        let key = format!("cache_test_{}", i);
        let data = vec![i as u8; 1024];
        cache_manager.set(key, data, None);
    }
    
    // 测试缓存访问
    let cache_hit_start = Instant::now();
    let mut cache_hits = 0;
    for _ in 0..100 {
        for i in 0..10 {
            let key = format!("cache_test_{}", i);
            if cache_manager.get(&key).is_some() {
                cache_hits += 1;
            }
        }
    }
    let cache_hit_time = cache_hit_start.elapsed();
    
    let cache_stats = cache_manager.get_stats();
    println!("   • 缓存命中: {} 次", cache_hits);
    println!("   • 命中率: {:.2}%", cache_stats.hit_rate * 100.0);
    println!("   • 平均访问时间: {:.2}μs", 
            cache_hit_time.as_micros() as f64 / (cache_hits as f64));
    
    Ok(())
}

// 辅助函数

/// 打印内存优化摘要
#[allow(dead_code)]
#[allow(unused_variables)]
async fn print_memory_optimization_summary(engine: &AIEngine) {
    let memory_stats = engine.get_memory_stats();
    
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│                   内存优化摘要                        │");
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│ 内存池统计:                                           │");
    println!("│ • 总块数: {:<42} │", memory_stats.pool_stats.total_blocks);
    println!("│ • 已分配块数: {:<38} │", memory_stats.pool_stats.allocated_blocks);
    println!("│ • 空闲块数: {:<40} │", memory_stats.pool_stats.free_blocks);
    println!("│ • 总内存: {:<43} │", format!("{} bytes", memory_stats.pool_stats.total_memory));
    println!("│ • 已用内存: {:<41} │", format!("{} bytes", memory_stats.pool_stats.used_memory));
    println!("│ • 分配次数: {:<41} │", memory_stats.pool_stats.allocation_count);
    println!("│ • 缓存命中: {:<41} │", memory_stats.pool_stats.cache_hits);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│ 缓存统计:                                             │");
    println!("│ • 总条目数: {:<40} │", memory_stats.cache_stats.total_entries);
    println!("│ • 总大小: {:<43} │", format!("{} bytes", memory_stats.cache_stats.total_size));
    println!("│ • 命中率: {:<44} │", format!("{:.2}%", memory_stats.cache_stats.hit_rate * 100.0));
    println!("│ • 驱逐次数: {:<41} │", memory_stats.cache_stats.eviction_count);
    println!("├─────────────────────────────────────────────────────────┤");
    println!("│ 零拷贝统计:                                           │");
    println!("│ • 缓冲区数量: {:<38} │", memory_stats.zero_copy_count);
    println!("│ • 节省内存: {:<41} │", format!("{} bytes", memory_stats.total_memory_saved));
    println!("│ • 优化比例: {:<41} │", format!("{:.2}%", memory_stats.optimization_ratio * 100.0));
    println!("└─────────────────────────────────────────────────────────┘");
    
    println!("\n🎯 内存优化特点:");
    println!("   • ✅ 内存池管理 - 减少内存分配开销");
    println!("   • ✅ 零拷贝缓冲区 - 避免不必要的数据复制");
    println!("   • ✅ 智能缓存系统 - LRU策略和TTL支持");
    println!("   • ✅ 自动内存清理 - 定期清理过期数据");
    println!("   • ✅ 内存使用监控 - 实时统计和优化建议");
    println!("   • ✅ 性能优化 - 显著提升内存操作效率");
    println!("   • ✅ 资源管理 - 智能内存回收和重用");
    println!("   • ✅ 压力测试 - 验证内存优化效果");
}
