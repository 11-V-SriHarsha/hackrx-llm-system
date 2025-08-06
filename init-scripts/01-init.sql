-- Initialize HackRx Database
-- This script runs automatically when PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
-- Note: Tables will be created by SQLAlchemy, but we can add indexes here

-- Grant additional permissions
GRANT ALL ON SCHEMA public TO hackrx_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO hackrx_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO hackrx_user;

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO hackrx_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO hackrx_user;

-- Create a simple test table to verify connection
CREATE TABLE IF NOT EXISTS connection_test (
    id SERIAL PRIMARY KEY,
    message TEXT DEFAULT 'PostgreSQL is working!',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO connection_test (message) VALUES ('Database initialized successfully!');