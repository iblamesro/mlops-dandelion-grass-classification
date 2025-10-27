-- Create plants_data table
CREATE TABLE IF NOT EXISTS plants_data (
    id SERIAL PRIMARY KEY,
    url_source VARCHAR(500) NOT NULL,
    url_s3 VARCHAR(500),
    label VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    file_size INTEGER,
    image_width INTEGER,
    image_height INTEGER
);

-- Create index for faster queries
CREATE INDEX idx_label ON plants_data(label);
CREATE INDEX idx_processed ON plants_data(processed);

-- Create model_metrics table for tracking
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT,
    precision_class FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    training_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table for monitoring
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    image_url VARCHAR(500),
    predicted_label VARCHAR(50),
    confidence FLOAT,
    model_version VARCHAR(50),
    prediction_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_date ON predictions(created_at);
