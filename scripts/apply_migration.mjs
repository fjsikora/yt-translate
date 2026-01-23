#!/usr/bin/env node
/**
 * Apply SQL migration to Supabase database using the Management API.
 */

import { createClient } from '@supabase/supabase-js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from .env file
const envPath = join(__dirname, '..', '.env');
const envContent = readFileSync(envPath, 'utf-8');
const env = {};
for (const line of envContent.split('\n')) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith('#') && trimmed.includes('=')) {
        const [key, ...valueParts] = trimmed.split('=');
        env[key.trim()] = valueParts.join('=').trim();
    }
}

const SUPABASE_URL = env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error('Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env');
    process.exit(1);
}

// Read the migration SQL
const migrationPath = join(__dirname, '..', 'supabase_migrations', 'migrations', '002_dubbing_studio_schema.sql');
const sql = readFileSync(migrationPath, 'utf-8');

console.log('Migration file:', migrationPath);
console.log('SQL size:', sql.length, 'bytes');

// Create Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// The Supabase JS client doesn't support raw SQL execution
// We need to use the postgres.js library directly or the REST API

console.log('\nNote: The Supabase JS client does not support raw SQL execution.');
console.log('The migration SQL file has been created at:');
console.log(' ', migrationPath);
console.log('\nTo apply this migration, you can:');
console.log('1. Copy the SQL and paste it in Supabase Dashboard > SQL Editor');
console.log('2. Use the Supabase CLI with: supabase db execute < migration.sql');
console.log('3. Connect directly with psql using the connection string from Supabase Dashboard');

// Let's try to verify if tables exist
console.log('\nChecking if tables already exist...');

async function checkTables() {
    const tables = ['dub_projects', 'dub_tracks', 'dub_segments'];

    for (const table of tables) {
        try {
            const { data, error } = await supabase.from(table).select('id').limit(1);
            if (error) {
                console.log(`✗ ${table}: ${error.message}`);
            } else {
                console.log(`✓ ${table} exists`);
            }
        } catch (e) {
            console.log(`✗ ${table}: ${e.message}`);
        }
    }
}

checkTables().then(() => {
    console.log('\nDone.');
});
