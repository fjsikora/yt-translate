#!/usr/bin/env node
/**
 * Apply SQL migration directly to Supabase PostgreSQL database.
 *
 * Requires the database password from Supabase Dashboard > Settings > Database
 * Set it as SUPABASE_DB_PASSWORD environment variable.
 */

import postgres from 'postgres';
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

// Database connection details
// Get from Supabase Dashboard > Settings > Database > Connection string
const PROJECT_REF = 'your-project-ref';
const DB_PASSWORD = env.SUPABASE_DB_PASSWORD || process.env.SUPABASE_DB_PASSWORD;

if (!DB_PASSWORD) {
    console.error('Error: SUPABASE_DB_PASSWORD not set');
    console.error('');
    console.error('To get the database password:');
    console.error('1. Go to Supabase Dashboard > Settings > Database');
    console.error('2. Find "Database password" section');
    console.error('3. Set it as: export SUPABASE_DB_PASSWORD=your_password');
    console.error('');
    console.error('Alternatively, add SUPABASE_DB_PASSWORD=your_password to .env');
    process.exit(1);
}

// Read the migration SQL
const migrationPath = join(__dirname, '..', 'supabase_migrations', 'migrations', '002_dubbing_studio_schema.sql');
const sql = readFileSync(migrationPath, 'utf-8');

console.log('Migration file:', migrationPath);
console.log('SQL size:', sql.length, 'bytes');

// Connect to database using Transaction Pooler (port 6543)
// Mode: transaction for best compatibility
const connectionString = `postgresql://postgres.${PROJECT_REF}:${DB_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:6543/postgres`;

async function applyMigration() {
    console.log('\nConnecting to Supabase PostgreSQL...');

    const client = postgres(connectionString, {
        ssl: { rejectUnauthorized: false },
        connection: {
            application_name: 'migration_script'
        }
    });

    try {
        // Execute the migration
        console.log('Executing migration...');
        await client.unsafe(sql);
        console.log('Migration executed successfully!');

        // Verify tables were created
        console.log('\nVerifying tables...');
        const tables = await client`
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'dub_%'
            ORDER BY table_name
        `;

        console.log('Created tables:', tables.map(t => t.table_name));

        // Count indexes
        const indexes = await client`
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname LIKE 'idx_dub_%'
        `;

        console.log('Created indexes:', indexes.length);

        return true;
    } catch (error) {
        console.error('Migration failed:', error.message);
        return false;
    } finally {
        await client.end();
    }
}

applyMigration().then(success => {
    process.exit(success ? 0 : 1);
});
