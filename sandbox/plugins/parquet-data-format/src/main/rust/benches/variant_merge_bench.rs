/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! V-33: merge-time reshredding cost vs the merge baseline.
//!
//! Two 100k-row segment files, one `id` (Int64, sort key) + one Variant column `v` built from a
//! CloudTrail-like corpus (~1.5 KB docs, copied from
//! `analytics-backend-datafusion/.../variant_poc3_smoke.rs`). Cells:
//!
//! - (a)  baseline merge, both segments unshredded, identical layout (existing merge path).
//! - (a2) baseline merge, both segments shredded on the same 6 paths, identical layout.
//! - (b)  unshredded -> shredded: `shred_variant` (6 paths) as a standalone pass over the merged
//!        200k-row stream read back from the (a) output file. The merge path cannot host it.
//! - (c)  shredded(A: 6 paths) + shredded(B: 3 different paths) -> C (6 paths):
//!        `unshred_variant` then `shred_variant` per batch, standalone pass over both inputs.
//! - (d)  leaf reuse: not implemented.
//! - mixed-layout merge: runs the existing merge path on unshredded + shredded inputs (both
//!        orders, sorted and unsorted paths) and prints the exact error.
//!
//! Median of 3 runs per cell. Wall-clock, us/row over the 200k merged rows, peak RSS (sampled
//! from /proc/self/statm every 2 ms), and the merge pool peak where available.
//!
//! Note: the merge output writer is rate limited to 20 MB/s (`io_task::RATE_LIMIT_MB_PER_SEC`,
//! hard-coded). The IO floor (output bytes / 20 MB/s) is printed next to every merge cell so
//! the CPU share can be read off.
//!
//! Run (from the workspace root sandbox/libs/dataformat-native/rust):
//!   cargo bench -p opensearch-parquet-format --bench variant_merge_bench
//! Env: VARIANT_MERGE_ROWS (rows per segment, default 100000), VARIANT_MERGE_RUNS (default 3).

use std::fs::{self, File};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::variant::{
    json_to_variant, shred_variant, unshred_variant, ShreddedSchemaBuilder, VariantArray,
};

use opensearch_parquet_format::merge::{merge_sorted, merge_unsorted, MergeError, MergeOutput};

const WRITE_BATCH: usize = 8192;
const READ_BATCH: usize = 8192;

// ── Corpus (copied from variant_poc3_smoke.rs) ─────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() as usize) % xs.len()]
    }
}

fn cloudtrail_doc(i: usize, rng: &mut Lcg) -> String {
    let events = [
        "GetObject",
        "PutObject",
        "ListObjects",
        "DeleteObject",
        "HeadObject",
        "CopyObject",
        "GetBucketAcl",
        "PutObjectTagging",
    ];
    let regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2"];
    let teams = ["search", "indexing", "storage", "platform", "security"];
    let envs = ["prod", "beta", "gamma"];
    let status = match rng.next() % 100 {
        0..=89 => 200,
        90..=95 => 403,
        96..=98 => 404,
        _ => 500,
    };
    let acct = 100_000_000_000u64 + (rng.next() % 900_000_000_000);
    let bucket = rng.next() % 5000;
    let obj = rng.next() % 1_000_000;
    format!(
        concat!(
            "{{\"eventVersion\":\"1.08\",\"eventTime\":\"2026-09-24T{:02}:{:02}:{:02}Z\",",
            "\"eventSource\":\"s3.amazonaws.com\",\"eventName\":\"{}\",\"awsRegion\":\"{}\",",
            "\"sourceIPAddress\":\"10.{}.{}.{}\",\"userAgent\":\"aws-cli/2.17.{} Python/3.12 Linux/5.10 exe/x86_64\",",
            "\"userIdentity\":{{\"type\":\"AssumedRole\",\"principalId\":\"AROA{:016X}:session-{}\",",
            "\"arn\":\"arn:aws:sts::{}:assumed-role/DataRole/session-{}\",\"accountId\":\"{}\",",
            "\"sessionContext\":{{\"attributes\":{{\"mfaAuthenticated\":\"{}\",\"creationDate\":\"2026-09-24T09:00:00Z\"}},",
            "\"sessionIssuer\":{{\"type\":\"Role\",\"principalId\":\"AROA{:016X}\",\"arn\":\"arn:aws:iam::{}:role/DataRole\",\"accountId\":\"{}\",\"userName\":\"DataRole\"}}}}}},",
            "\"requestParameters\":{{\"bucketName\":\"bucket-{}\",\"key\":\"data/dt=2026-09-24/part-{:07}.parquet\",",
            "\"Host\":\"bucket-{}.s3.amazonaws.com\",\"x-amz-acl\":\"private\",",
            "\"tags\":[{{\"key\":\"team\",\"value\":\"{}\"}},{{\"key\":\"env\",\"value\":\"{}\"}}]}},",
            "\"responseElements\":{{\"statusCode\":{},\"x-amz-request-id\":\"{:016X}\",\"x-amz-id-2\":\"{:016X}{:016X}\"}},",
            "\"requestID\":\"{:016X}\",\"eventID\":\"{:08x}-{:04x}-{:04x}-{:04x}-{:012x}\",\"readOnly\":{},",
            "\"resources\":[{{\"type\":\"AWS::S3::Object\",\"ARN\":\"arn:aws:s3:::bucket-{}/data/part-{:07}.parquet\"}},",
            "{{\"type\":\"AWS::S3::Bucket\",\"ARN\":\"arn:aws:s3:::bucket-{}\",\"accountId\":\"{}\"}}],",
            "\"eventType\":\"AwsApiCall\",\"managementEvent\":false,\"recipientAccountId\":\"{}\",\"eventCategory\":\"Data\",",
            "\"tlsDetails\":{{\"tlsVersion\":\"TLSv1.3\",\"cipherSuite\":\"TLS_AES_128_GCM_SHA256\",\"clientProvidedHostHeader\":\"bucket-{}.s3.amazonaws.com\"}}}}"
        ),
        (i / 3600) % 24, (i / 60) % 60, i % 60,
        rng.pick(&events), rng.pick(&regions),
        rng.next() % 256, rng.next() % 256, rng.next() % 256, rng.next() % 60,
        rng.next(), i % 1000, acct, i % 1000, acct,
        if rng.next() % 3 == 0 { "true" } else { "false" },
        rng.next(), acct, acct,
        bucket, obj, bucket,
        rng.pick(&teams), rng.pick(&envs),
        status, rng.next(), rng.next(), rng.next(),
        rng.next(), rng.next() as u32, rng.next() as u16, rng.next() as u16, rng.next() as u16, rng.next() & 0xffff_ffff_ffff,
        if status == 200 && rng.next() % 2 == 0 { "true" } else { "false" },
        bucket, obj, bucket, acct, acct, bucket
    )
}

fn corpus(seg: usize, n: usize) -> ArrayRef {
    let mut rng = Lcg(0x5eed ^ (seg as u64 * 0x9e37_79b9));
    let docs: Vec<String> = (0..n)
        .map(|i| cloudtrail_doc(seg * n + i, &mut rng))
        .collect();
    Arc::new(StringArray::from(docs))
}

/// The 6-path shape from variant_poc3_smoke.rs (the "current" shredding schema, C).
fn shape_6() -> DataType {
    ShreddedSchemaBuilder::new()
        .with_path("responseElements.statusCode", (&DataType::Int64, true))
        .unwrap()
        .with_path("eventName", (&DataType::Utf8, true))
        .unwrap()
        .with_path("eventTime", (&DataType::Utf8, true))
        .unwrap()
        .with_path("awsRegion", (&DataType::Utf8, true))
        .unwrap()
        .with_path("readOnly", (&DataType::Boolean, true))
        .unwrap()
        .with_path("userIdentity.accountId", (&DataType::Utf8, true))
        .unwrap()
        .build()
}

/// A different 3-path shape (an "old" schema, B) that shares no path with `shape_6`.
fn shape_3() -> DataType {
    ShreddedSchemaBuilder::new()
        .with_path("eventSource", (&DataType::Utf8, true))
        .unwrap()
        .with_path("sourceIPAddress", (&DataType::Utf8, true))
        .unwrap()
        .with_path("requestParameters.bucketName", (&DataType::Utf8, true))
        .unwrap()
        .build()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Unshredded,
    Shredded6,
    Shredded3,
}

// ── Segment files ───────────────────────────────────────────────────────────

/// Builds one segment: ids interleave across segments (`2*i + seg`) so the sorted merge has to
/// actually alternate between cursors. Returns the file path and its size in bytes.
fn write_segment(path: &str, seg: usize, rows: usize, layout: Layout) -> u64 {
    let text = corpus(seg, rows);
    let unshredded = json_to_variant(&text).unwrap();
    let variant = match layout {
        Layout::Unshredded => unshredded,
        Layout::Shredded6 => shred_variant(&unshredded, &shape_6()).unwrap(),
        Layout::Shredded3 => shred_variant(&unshredded, &shape_3()).unwrap(),
    };
    let v_field = variant.field("v");
    let v_arr: ArrayRef = Arc::new(variant.into_inner());
    let ids = Int64Array::from_iter_values((0..rows).map(|i| (2 * i + seg) as i64));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        v_field,
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(rows))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema.clone(), Some(props))
        .unwrap();
    let mut off = 0;
    while off < rows {
        let len = WRITE_BATCH.min(rows - off);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids.slice(off, len)), v_arr.slice(off, len)],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        off += len;
    }
    writer.close().unwrap();
    fs::metadata(path).unwrap().len()
}

/// Reads the `v` column of a parquet file as a list of batches (READ_BATCH rows each).
fn read_v_column(path: &str) -> Vec<ArrayRef> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
    let v_idx = builder.schema().index_of("v").unwrap();
    let mask = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), [v_idx]);
    let reader = builder
        .with_batch_size(READ_BATCH)
        .with_projection(mask)
        .build()
        .unwrap();
    reader
        .map(|b| b.unwrap().column(0).clone())
        .collect()
}

// ── Measurement helpers ─────────────────────────────────────────────────────

fn rss_bytes() -> usize {
    let statm = fs::read_to_string("/proc/self/statm").unwrap_or_default();
    statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
        * 4096
}

/// Samples /proc/self/statm on a background thread and records the peak RSS.
struct RssSampler {
    peak: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start() -> Self {
        let peak = Arc::new(AtomicUsize::new(rss_bytes()));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let handle = thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(rss_bytes(), Ordering::Relaxed);
                thread::sleep(Duration::from_millis(2));
            }
        });
        Self {
            peak,
            stop,
            handle: Some(handle),
        }
    }
    fn stop(mut self) -> usize {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peak.load(Ordering::Relaxed)
    }
}

struct Sample {
    wall: Duration,
    peak_rss: usize,
    /// Merge-only: flush+encode millis reported by the merge path, output bytes, pool peak.
    merge: Option<(i64, u64, usize)>,
}

fn median(mut xs: Vec<Sample>) -> Sample {
    xs.sort_by_key(|s| s.wall);
    xs.swap_remove(xs.len() / 2)
}

fn timed<F: FnMut() -> Option<(i64, u64, usize)>>(mut f: F) -> Sample {
    let sampler = RssSampler::start();
    let t = Instant::now();
    let merge = f();
    let wall = t.elapsed();
    let peak_rss = sampler.stop();
    Sample {
        wall,
        peak_rss,
        merge,
    }
}

fn mb(b: usize) -> f64 {
    b as f64 / 1024.0 / 1024.0
}

struct Row {
    cell: String,
    rows: usize,
    s: Sample,
}

impl Row {
    fn us_per_row(&self) -> f64 {
        self.s.wall.as_secs_f64() * 1e6 / self.rows as f64
    }
}

// ── Cells ───────────────────────────────────────────────────────────────────

fn merge_pool_peak() -> usize {
    opensearch_parquet_format::memory::get_stats()[5]
}

fn run_merge(inputs: &[String], output: &str, sorted: bool) -> Result<MergeOutput, MergeError> {
    if sorted {
        merge_sorted(
            inputs,
            output,
            "variant_merge_bench",
            &["id".to_string()],
            &[false],
            &[false],
            1,
        )
    } else {
        merge_unsorted(inputs, output, "variant_merge_bench", 1)
    }
}

fn merge_cell(label: &str, inputs: &[String], output: &str, runs: usize, total_rows: usize) -> Row {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let _ = fs::remove_file(output);
        let s = timed(|| {
            let out = run_merge(inputs, output, true).unwrap_or_else(|e| {
                panic!("{label}: merge failed: {e}");
            });
            assert_eq!(out.metadata.file_metadata().num_rows() as usize, total_rows);
            let bytes = fs::metadata(output).map(|m| m.len()).unwrap_or(0);
            Some((out.flush_and_sort_chunk_time_millis, bytes, merge_pool_peak()))
        });
        samples.push(s);
    }
    Row {
        cell: label.to_string(),
        rows: total_rows,
        s: median(samples),
    }
}

/// (b): shred_variant over already-read unshredded batches.
fn shred_cell(label: &str, batches: &[ArrayRef], shape: &DataType, runs: usize) -> Row {
    let rows: usize = batches.iter().map(|b| b.len()).sum();
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        samples.push(timed(|| {
            let mut out_rows = 0usize;
            for b in batches {
                let va = VariantArray::try_new(b.as_ref()).unwrap();
                let sh = shred_variant(&va, shape).unwrap();
                out_rows += sh.len();
                assert!(sh.typed_value_field().is_some());
            }
            assert_eq!(out_rows, rows);
            None
        }));
    }
    Row {
        cell: label.to_string(),
        rows,
        s: median(samples),
    }
}

/// (c): unshred_variant then shred_variant over already-read shredded batches.
fn reshred_cell(label: &str, batches: &[ArrayRef], shape: &DataType, runs: usize) -> Row {
    let rows: usize = batches.iter().map(|b| b.len()).sum();
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        samples.push(timed(|| {
            let mut out_rows = 0usize;
            for b in batches {
                let va = VariantArray::try_new(b.as_ref()).unwrap();
                let un = unshred_variant(&va).unwrap();
                assert!(un.typed_value_field().is_none());
                let sh = shred_variant(&un, shape).unwrap();
                out_rows += sh.len();
            }
            assert_eq!(out_rows, rows);
            None
        }));
    }
    Row {
        cell: label.to_string(),
        rows,
        s: median(samples),
    }
}

fn unshred_only_cell(label: &str, batches: &[ArrayRef], runs: usize) -> Row {
    let rows: usize = batches.iter().map(|b| b.len()).sum();
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        samples.push(timed(|| {
            for b in batches {
                let va = VariantArray::try_new(b.as_ref()).unwrap();
                let un = unshred_variant(&va).unwrap();
                assert_eq!(un.len(), b.len());
            }
            None
        }));
    }
    Row {
        cell: label.to_string(),
        rows,
        s: median(samples),
    }
}

fn read_cell(label: &str, paths: &[String], runs: usize) -> (Row, Vec<ArrayRef>) {
    let mut samples = Vec::with_capacity(runs);
    let mut batches = Vec::new();
    for _ in 0..runs {
        samples.push(timed(|| {
            batches = paths.iter().flat_map(|p| read_v_column(p)).collect();
            None
        }));
    }
    let rows = batches.iter().map(|b| b.len()).sum();
    (
        Row {
            cell: label.to_string(),
            rows,
            s: median(samples),
        },
        batches,
    )
}

fn classify(e: &MergeError) -> &'static str {
    match e {
        MergeError::Logic(s) if s.starts_with("Failed to compute union schema") => {
            "merge/context.rs MergeContext::new -> arrow Schema::try_merge"
        }
        MergeError::Logic(_) => "merge (Logic)",
        MergeError::Arrow(_) => {
            "merge/schema.rs append_row_id -> RecordBatch::try_new, called from merge/context.rs \
             MergeContext::push_batch (ColumnMapping::pad_batch short-circuits on is_identity and \
             never type-checks the struct)"
        }
        MergeError::Parquet(_) => {
            "merge/context.rs MergeContext::new/push_batch -> parquet writer (compute_leaves / ArrowRowGroupWriterFactory)"
        }
        MergeError::Io(_) => "IO",
    }
}

static LAST_PANIC: std::sync::Mutex<Option<(String, String)>> = std::sync::Mutex::new(None);

fn install_panic_capture() {
    std::panic::set_hook(Box::new(|info| {
        let msg = match info.payload().downcast_ref::<&str>() {
            Some(s) => s.to_string(),
            None => match info.payload().downcast_ref::<String>() {
                Some(s) => s.clone(),
                None => "<non-string panic payload>".to_string(),
            },
        };
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        // Keep only frames from our crate and the parquet/arrow crates, in order.
        let mut frames = Vec::new();
        let mut lines = bt.lines().peekable();
        while let Some(l) = lines.next() {
            let t = l.trim();
            if t.contains("opensearch_parquet_format::")
                || t.contains("parquet::")
                || t.contains("arrow_array::")
                || t.contains("variant_merge_bench::")
            {
                let mut f = t.to_string();
                if let Some(at) = lines.peek() {
                    if at.trim().starts_with("at ") {
                        f.push_str("  ");
                        f.push_str(at.trim());
                    }
                }
                frames.push(f);
            }
            if frames.len() >= 14 {
                break;
            }
        }
        *LAST_PANIC.lock().unwrap() = Some((format!("{msg} (at {loc})"), frames.join("\n      ")));
    }));
}

fn mixed_layout_probe(label: &str, inputs: &[String], output: &str, sorted: bool) {
    let _ = fs::remove_file(output);
    let path = if sorted { "merge_sorted" } else { "merge_unsorted" };
    *LAST_PANIC.lock().unwrap() = None;
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_merge(inputs, output, sorted)
    }));
    match res {
        Ok(Ok(out)) => println!(
            "  {label} [{path}]: UNEXPECTEDLY SUCCEEDED ({} rows written)",
            out.metadata.file_metadata().num_rows()
        ),
        Ok(Err(e)) => {
            println!("  {label} [{path}]: FAILED with Err");
            println!("    error variant: {}", match &e {
                MergeError::Arrow(_) => "MergeError::Arrow",
                MergeError::Parquet(_) => "MergeError::Parquet",
                MergeError::Io(_) => "MergeError::Io",
                MergeError::Logic(_) => "MergeError::Logic",
            });
            println!("    message:       {e}");
            println!("    frame:         {}", classify(&e));
        }
        Err(_) => {
            println!("  {label} [{path}]: PANICKED");
            if let Some((msg, frames)) = LAST_PANIC.lock().unwrap().clone() {
                println!("    panic:         {msg}");
                println!("    frames:\n      {frames}");
            }
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let rows: usize = std::env::var("VARIANT_MERGE_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let runs: usize = std::env::var("VARIANT_MERGE_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let total = 2 * rows;
    let profile = if cfg!(debug_assertions) { "dev" } else { "release" };

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(" V-33 Variant merge benchmark: 2 x {rows} rows, median of {runs}, profile {profile}");
    println!("═══════════════════════════════════════════════════════════════════════");

    let dir = tempfile::tempdir().unwrap();
    let p = |name: &str| dir.path().join(name).to_str().unwrap().to_string();

    // ── Inputs ──
    let t0 = Instant::now();
    let un = [p("un_0.parquet"), p("un_1.parquet")];
    let sh6 = [p("sh6_0.parquet"), p("sh6_1.parquet")];
    let sh3_1 = p("sh3_1.parquet");
    let mut sizes = Vec::new();
    for (seg, path) in un.iter().enumerate() {
        sizes.push((format!("seg{seg} unshredded"), write_segment(path, seg, rows, Layout::Unshredded)));
    }
    for (seg, path) in sh6.iter().enumerate() {
        sizes.push((format!("seg{seg} shredded(6)"), write_segment(path, seg, rows, Layout::Shredded6)));
    }
    sizes.push(("seg1 shredded(3)".into(), write_segment(&sh3_1, 1, rows, Layout::Shredded3)));
    println!("\nInputs written in {:.1?}:", t0.elapsed());
    for (l, b) in &sizes {
        println!("  {l:<20} {:>8.1} MB  ({:.0} B/row)", mb(*b as usize), *b as f64 / rows as f64);
    }
    {
        // Print the Parquet schema of one unshredded and one shredded input for the record.
        for (l, path) in [("unshredded", &un[0]), ("shredded(6)", &sh6[0]), ("shredded(3)", &sh3_1)] {
            let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
            let f = b.schema().field_with_name("v").unwrap();
            println!("  {l:<12} v: {}  ext={:?}", f.data_type(), f.metadata().get("ARROW:extension:name"));
        }
    }

    let mut table: Vec<Row> = Vec::new();

    // ── (a) baseline: unshredded + unshredded ──
    let out_a = p("out_a.parquet");
    table.push(merge_cell("(a)  merge un+un -> un (sorted)", &un, &out_a, runs, total));
    let out_a_uns = p("out_a_unsorted.parquet");
    table.push(merge_cell("(a') merge un+un -> un (unsorted)", &un, &out_a_uns, runs, total));

    // ── (a2) baseline: shredded(6) + shredded(6) ──
    let out_a2 = p("out_a2.parquet");
    table.push(merge_cell("(a2) merge sh6+sh6 -> sh6 (sorted)", &sh6, &out_a2, runs, total));

    // ── (b) unshredded merged stream -> shredded(6): standalone pass over the (a) output ──
    let (read_a, un_batches) = read_cell("read  (a) output `v` (200k rows, parquet -> arrow)", &[out_a.clone()], runs);
    table.push(read_a);
    table.push(shred_cell("(b)  shred_variant(6) over merged un stream", &un_batches, &shape_6(), runs));
    drop(un_batches);

    // ── (c) shredded(6) A + shredded(3) B -> shredded(6): unshred + shred, standalone ──
    let mixed_inputs = [sh6[0].clone(), sh3_1.clone()];
    let (read_c, mixed_batches) = read_cell("read  A(sh6)+B(sh3) `v` (200k rows)", &mixed_inputs, runs);
    table.push(read_c);
    table.push(unshred_only_cell("(c0) unshred_variant only over A+B", &mixed_batches, runs));
    table.push(reshred_cell("(c)  unshred+shred(6) over A(sh6)+B(sh3)", &mixed_batches, &shape_6(), runs));
    // Per-layout split so the A (6->6) and B (3->6) halves can be told apart.
    let a_only: Vec<ArrayRef> = read_v_column(&sh6[0]);
    let b_only: Vec<ArrayRef> = read_v_column(&sh3_1);
    table.push(reshred_cell("(c-A) unshred+shred(6) over A(sh6) only", &a_only, &shape_6(), runs));
    table.push(reshred_cell("(c-B) unshred+shred(6) over B(sh3) only", &b_only, &shape_6(), runs));
    drop((mixed_batches, a_only, b_only));

    // ── Table ──
    println!();
    println!(
        "{:<52} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "cell", "us/row", "wall", "peakRSS", "flushMs", "outMB", "IOfloor"
    );
    println!("{}", "-".repeat(118));
    for r in &table {
        let (flush, outmb, io_floor) = match r.s.merge {
            Some((ms, bytes, _pool)) => (
                format!("{ms}"),
                format!("{:.1}", mb(bytes as usize)),
                format!("{:.2}s", bytes as f64 / (20.0 * 1024.0 * 1024.0)),
            ),
            None => ("-".into(), "-".into(), "-".into()),
        };
        println!(
            "{:<52} {:>9.2} {:>10.2?} {:>8.0}MB {:>10} {:>10} {:>10}",
            r.cell,
            r.us_per_row(),
            r.s.wall,
            mb(r.s.peak_rss),
            flush,
            outmb,
            io_floor
        );
    }
    println!("{}", "-".repeat(118));
    println!("IOfloor = output bytes / 20 MB/s (io_task::RATE_LIMIT_MB_PER_SEC, hard-coded on the merge output writer).");
    println!("flushMs = MergeOutput.flush_and_sort_chunk_time_millis (column close: encode+compress, CPU).");
    if let Some(pool) = table.iter().filter_map(|r| r.s.merge.map(|m| m.2)).max() {
        println!("merge pool peak (native_bridge_common MemoryPool, tracked bytes): {:.0} MB", mb(pool));
    }

    // ── Ratios ──
    let find = |prefix: &str| table.iter().find(|r| r.cell.starts_with(prefix)).unwrap();
    let a = find("(a)  ");
    let a2 = find("(a2)");
    let b = find("(b)");
    let c = find("(c)  ");
    let a_cpu = a.s.wall.as_secs_f64() - a.s.merge.map(|m| m.1 as f64 / (20.0 * 1024.0 * 1024.0)).unwrap_or(0.0);
    let a_cpu_us = a_cpu * 1e6 / a.rows as f64;
    println!();
    println!("Ratios (per row, over {} merged rows):", total);
    println!(
        "  (a) baseline merge:            {:>7.2} us/row wall   (~{:.2} us/row after subtracting the 20 MB/s IO floor)",
        a.us_per_row(),
        a_cpu_us
    );
    println!("  (a2) baseline merge shredded:  {:>7.2} us/row wall", a2.us_per_row());
    println!(
        "  (b) shred at merge:            {:>7.2} us/row  = +{:.0}% / {:.2}x of (a) wall, +{:.0}% / {:.2}x of (a) CPU-only",
        b.us_per_row(),
        100.0 * b.us_per_row() / a.us_per_row(),
        b.us_per_row() / a.us_per_row(),
        100.0 * b.us_per_row() / a_cpu_us,
        b.us_per_row() / a_cpu_us
    );
    println!(
        "  (c) unshred+reshred at merge:  {:>7.2} us/row  = +{:.0}% / {:.2}x of (a) wall, +{:.0}% / {:.2}x of (a) CPU-only",
        c.us_per_row(),
        100.0 * c.us_per_row() / a.us_per_row(),
        c.us_per_row() / a.us_per_row(),
        100.0 * c.us_per_row() / a_cpu_us,
        c.us_per_row() / a_cpu_us
    );
    println!("  (d) leaf reuse: not implemented.");

    // ── Mixed-layout merge through the existing path ──
    println!();
    println!("Mixed-layout merge through the existing merge path (expected to fail):");
    install_panic_capture();
    let out_m = p("out_mixed.parquet");
    mixed_layout_probe("un(seg0) + sh6(seg1)", &[un[0].clone(), sh6[1].clone()], &out_m, true);
    mixed_layout_probe("sh6(seg0) + un(seg1)", &[sh6[0].clone(), un[1].clone()], &out_m, true);
    mixed_layout_probe("sh6(seg0) + sh3(seg1)", &[sh6[0].clone(), sh3_1.clone()], &out_m, true);
    mixed_layout_probe("un(seg0) + sh6(seg1)", &[un[0].clone(), sh6[1].clone()], &out_m, false);
    mixed_layout_probe("sh6(seg0) + un(seg1)", &[sh6[0].clone(), un[1].clone()], &out_m, false);
    println!();
}
