-- Run once before deploying semantic-section ingestion.
ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS section_id UUID;
ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS record_type TEXT NOT NULL DEFAULT 'transcript';
ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS row_start INTEGER;
ALTER TABLE transcript_chunks ADD COLUMN IF NOT EXISTS row_end INTEGER;
CREATE INDEX IF NOT EXISTS transcript_chunks_section_id_idx ON transcript_chunks (section_id);
CREATE INDEX IF NOT EXISTS transcript_chunks_video_section_idx ON transcript_chunks (youtube_video_id, section_id);
