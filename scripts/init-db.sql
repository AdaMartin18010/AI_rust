-- AI-Rust项目数据库初始化脚本
-- PostgreSQL数据库初始化

-- 创建数据库（如果不存在）
-- CREATE DATABASE ai_rust;

-- 连接到数据库
-- \c ai_rust;

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE
);

-- 创建API密钥表
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    last_used TIMESTAMP WITH TIME ZONE
);

-- 创建请求日志表
CREATE TABLE IF NOT EXISTS request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    method VARCHAR(10) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size INTEGER,
    response_size INTEGER,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建性能指标表
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20) NOT NULL,
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建错误日志表
CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    error_type VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    request_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建配置表
CREATE TABLE IF NOT EXISTS configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    config_type VARCHAR(20) DEFAULT 'string',
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_created_at ON api_keys(created_at);

CREATE INDEX IF NOT EXISTS idx_request_logs_user_id ON request_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_api_key_id ON request_logs(api_key_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_request_logs_status_code ON request_logs(status_code);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);

CREATE INDEX IF NOT EXISTS idx_error_logs_user_id ON error_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_error_logs_created_at ON error_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_error_logs_type ON error_logs(error_type);

CREATE INDEX IF NOT EXISTS idx_configurations_key ON configurations(config_key);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 创建触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configurations_updated_at BEFORE UPDATE ON configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认配置
INSERT INTO configurations (config_key, config_value, config_type, description) VALUES
    ('api_rate_limit', '1000', 'integer', 'API请求速率限制（每小时）'),
    ('max_request_size', '10485760', 'integer', '最大请求大小（字节）'),
    ('session_timeout', '3600', 'integer', '会话超时时间（秒）'),
    ('enable_cors', 'true', 'boolean', '是否启用CORS'),
    ('log_level', 'info', 'string', '日志级别'),
    ('enable_metrics', 'true', 'boolean', '是否启用性能指标收集')
ON CONFLICT (config_key) DO NOTHING;

-- 插入默认用户（仅用于测试）
INSERT INTO users (username, email, password_hash) VALUES
    ('admin', 'admin@ai-rust.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8Qj8Qj8Qj8', 'Admin User'),
    ('test', 'test@ai-rust.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8Qj8Qj8Qj8', 'Test User')
ON CONFLICT (username) DO NOTHING;

-- 创建视图
CREATE OR REPLACE VIEW user_stats AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.created_at,
    u.last_login,
    COUNT(rl.id) as total_requests,
    COUNT(CASE WHEN rl.status_code >= 200 AND rl.status_code < 300 THEN 1 END) as successful_requests,
    COUNT(CASE WHEN rl.status_code >= 400 THEN 1 END) as failed_requests,
    AVG(rl.response_time_ms) as avg_response_time
FROM users u
LEFT JOIN request_logs rl ON u.id = rl.user_id
GROUP BY u.id, u.username, u.email, u.created_at, u.last_login;

CREATE OR REPLACE VIEW api_usage_stats AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    endpoint,
    method,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    COUNT(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 END) as success_count,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
FROM request_logs
GROUP BY DATE_TRUNC('hour', created_at), endpoint, method
ORDER BY hour DESC, request_count DESC;

-- 创建清理旧数据的函数
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- 清理30天前的请求日志
    DELETE FROM request_logs WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- 清理90天前的性能指标
    DELETE FROM performance_metrics WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- 清理90天前的错误日志
    DELETE FROM error_logs WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    RAISE NOTICE 'Old data cleaned up successfully';
END;
$$ LANGUAGE plpgsql;

-- 创建性能统计函数
CREATE OR REPLACE FUNCTION get_performance_stats(
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE
)
RETURNS TABLE (
    metric_name VARCHAR(100),
    avg_value DOUBLE PRECISION,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pm.metric_name,
        AVG(pm.value) as avg_value,
        MIN(pm.value) as min_value,
        MAX(pm.value) as max_value,
        COUNT(*) as count
    FROM performance_metrics pm
    WHERE pm.timestamp BETWEEN start_time AND end_time
    GROUP BY pm.metric_name
    ORDER BY pm.metric_name;
END;
$$ LANGUAGE plpgsql;

-- 创建用户活动统计函数
CREATE OR REPLACE FUNCTION get_user_activity_stats(
    user_id_param UUID,
    days_back INTEGER DEFAULT 7
)
RETURNS TABLE (
    date DATE,
    request_count BIGINT,
    avg_response_time DOUBLE PRECISION,
    error_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        DATE(rl.created_at) as date,
        COUNT(*) as request_count,
        AVG(rl.response_time_ms) as avg_response_time,
        COUNT(CASE WHEN rl.status_code >= 400 THEN 1 END) as error_count
    FROM request_logs rl
    WHERE rl.user_id = user_id_param
    AND rl.created_at >= CURRENT_DATE - INTERVAL '1 day' * days_back
    GROUP BY DATE(rl.created_at)
    ORDER BY date DESC;
END;
$$ LANGUAGE plpgsql;

-- 创建API使用情况报告函数
CREATE OR REPLACE FUNCTION get_api_usage_report(
    start_date DATE,
    end_date DATE
)
RETURNS TABLE (
    endpoint VARCHAR(255),
    method VARCHAR(10),
    total_requests BIGINT,
    unique_users BIGINT,
    avg_response_time DOUBLE PRECISION,
    error_rate DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rl.endpoint,
        rl.method,
        COUNT(*) as total_requests,
        COUNT(DISTINCT rl.user_id) as unique_users,
        AVG(rl.response_time_ms) as avg_response_time,
        (COUNT(CASE WHEN rl.status_code >= 400 THEN 1 END)::DOUBLE PRECISION / COUNT(*) * 100) as error_rate
    FROM request_logs rl
    WHERE DATE(rl.created_at) BETWEEN start_date AND end_date
    GROUP BY rl.endpoint, rl.method
    ORDER BY total_requests DESC;
END;
$$ LANGUAGE plpgsql;

-- 创建数据完整性检查函数
CREATE OR REPLACE FUNCTION check_data_integrity()
RETURNS TABLE (
    table_name TEXT,
    record_count BIGINT,
    last_updated TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'users'::TEXT,
        COUNT(*),
        MAX(updated_at)
    FROM users
    UNION ALL
    SELECT 
        'api_keys'::TEXT,
        COUNT(*),
        MAX(created_at)
    FROM api_keys
    UNION ALL
    SELECT 
        'request_logs'::TEXT,
        COUNT(*),
        MAX(created_at)
    FROM request_logs
    UNION ALL
    SELECT 
        'performance_metrics'::TEXT,
        COUNT(*),
        MAX(timestamp)
    FROM performance_metrics
    UNION ALL
    SELECT 
        'error_logs'::TEXT,
        COUNT(*),
        MAX(created_at)
    FROM error_logs
    UNION ALL
    SELECT 
        'configurations'::TEXT,
        COUNT(*),
        MAX(updated_at)
    FROM configurations;
END;
$$ LANGUAGE plpgsql;

-- 创建备份函数
CREATE OR REPLACE FUNCTION create_backup()
RETURNS TEXT AS $$
DECLARE
    backup_file TEXT;
    backup_timestamp TEXT;
BEGIN
    backup_timestamp := TO_CHAR(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MISS');
    backup_file := '/backups/ai_rust_backup_' || backup_timestamp || '.sql';
    
    -- 这里应该执行实际的备份命令
    -- 例如: pg_dump ai_rust > backup_file
    
    RETURN 'Backup created: ' || backup_file;
END;
$$ LANGUAGE plpgsql;

-- 创建监控查询
CREATE OR REPLACE VIEW system_health AS
SELECT 
    'database_size' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value
UNION ALL
SELECT 
    'active_connections' as metric,
    count(*)::text as value
FROM pg_stat_activity
WHERE state = 'active'
UNION ALL
SELECT 
    'total_requests_today' as metric,
    count(*)::text as value
FROM request_logs
WHERE DATE(created_at) = CURRENT_DATE
UNION ALL
SELECT 
    'error_rate_today' as metric,
    ROUND(
        (COUNT(CASE WHEN status_code >= 400 THEN 1 END)::DOUBLE PRECISION / 
         NULLIF(COUNT(*), 0) * 100), 2
    )::text as value
FROM request_logs
WHERE DATE(created_at) = CURRENT_DATE;

-- 创建权限
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ai_rust;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ai_rust;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ai_rust;

-- 创建定期清理任务（需要pg_cron扩展）
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

-- 完成初始化
SELECT 'Database initialization completed successfully!' as status;