/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! POC-3 smoke benchmark (CPU-only, in-memory): one workload (CloudTrail-like S3 data events,
//! ~35 keys, nested `userIdentity` / `requestParameters` / `responseElements`, two arrays), three
//! layouts (JSON text + `json_extract`, unshredded Variant + `variant_get`, Variant shredded on
//! six declared paths + `variant_get`), and the four Snowflake query shapes: field extraction,
//! full object, array element, full array. Also measures Rust-side encode (`json_to_variant`)
//! and flush-time shredding (`shred_variant`) cost per row, and in-memory sizes.
//!
//! This is the V-18 smoke, not the full POC-3 (no IO, no pruning, one workload, 100k rows).
//!
//! Run: `VARIANT_POC3_ROWS=100000 cargo test --release -p opensearch-datafusion variant_poc3 -- --ignored --nocapture`

#![cfg(test)]

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Array, ArrayRef, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::ScalarUDF;
use datafusion::physical_expr::expressions::{Column as PhysColumn, Literal};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use parquet::variant::{
    json_to_variant, shred_variant, variant_to_json, ShreddedSchemaBuilder, VariantArray,
};

use crate::udf::json_extract::JsonExtractUdf;
use crate::udf::variant_get::VariantGetUdf;

const BATCH: usize = 8192;
const RUNS: usize = 5;

// ── Corpus ──────────────────────────────────────────────────────────

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

fn corpus(n: usize) -> ArrayRef {
    let mut rng = Lcg(0x5eed);
    let docs: Vec<String> = (0..n).map(|i| cloudtrail_doc(i, &mut rng)).collect();
    Arc::new(StringArray::from(docs))
}

fn shredding_shape() -> DataType {
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

// ── Expressions ─────────────────────────────────────────────────────

fn lit(v: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Utf8(Some(v.to_string()))))
}

fn json_extract(schema: &Schema, path: &str) -> Arc<dyn PhysicalExpr> {
    let udf = Arc::new(ScalarUDF::from(JsonExtractUdf::new()));
    Arc::new(
        ScalarFunctionExpr::try_new(
            udf,
            vec![
                Arc::new(PhysColumn::new("t", schema.index_of("t").unwrap())),
                lit(path),
            ],
            schema,
            Arc::new(ConfigOptions::default()),
        )
        .unwrap(),
    )
}

fn variant_get(schema: &Schema, path: &str, ty: Option<&str>) -> Arc<dyn PhysicalExpr> {
    let udf = Arc::new(ScalarUDF::from(VariantGetUdf::new()));
    let mut args: Vec<Arc<dyn PhysicalExpr>> = vec![
        Arc::new(PhysColumn::new("v", schema.index_of("v").unwrap())),
        lit(path),
    ];
    if let Some(t) = ty {
        args.push(lit(t));
    }
    Arc::new(
        ScalarFunctionExpr::try_new(udf, args, schema, Arc::new(ConfigOptions::default())).unwrap(),
    )
}

// ── Timing ──────────────────────────────────────────────────────────

fn batches(schema: &SchemaRef, col: &ArrayRef) -> Vec<RecordBatch> {
    (0..col.len())
        .step_by(BATCH)
        .map(|off| {
            let len = BATCH.min(col.len() - off);
            RecordBatch::try_new(schema.clone(), vec![col.slice(off, len)]).unwrap()
        })
        .collect()
}

/// Median over RUNS of total evaluation time; returns ns per row and the non-null count from
/// the last run (sanity check that the expression actually extracted something).
fn time_expr(expr: &Arc<dyn PhysicalExpr>, batches: &[RecordBatch], rows: usize) -> (f64, usize) {
    let mut samples = Vec::with_capacity(RUNS);
    let mut non_null = 0usize;
    for _ in 0..RUNS {
        non_null = 0;
        let t = Instant::now();
        for b in batches {
            let out = expr.evaluate(b).unwrap().into_array(b.num_rows()).unwrap();
            non_null += out.len() - out.null_count();
        }
        samples.push(t.elapsed().as_nanos() as f64 / rows as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[RUNS / 2], non_null)
}

fn time_fn<F: FnMut()>(mut f: F, rows: usize) -> f64 {
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64 / rows as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[RUNS / 2]
}

#[test]
#[ignore]
fn variant_poc3_smoke() {
    let rows: usize = std::env::var("VARIANT_POC3_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let profile = if cfg!(debug_assertions) {
        "dev (opt-level 1, debug assertions)"
    } else {
        "release"
    };
    eprintln!("\n=== POC-3 smoke: {rows} CloudTrail-like docs, batch {BATCH}, median of {RUNS} runs, profile {profile} ===");

    let text = corpus(rows);
    let avg_len = text
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .flatten()
        .map(|s| s.len())
        .sum::<usize>()
        / rows;
    eprintln!("avg document length: {avg_len} bytes");

    // Layouts.
    let mut unshredded: Option<VariantArray> = None;
    let enc_ns = time_fn(|| unshredded = Some(json_to_variant(&text).unwrap()), rows);
    let unshredded = unshredded.unwrap();
    let shape = shredding_shape();
    let mut shredded: Option<VariantArray> = None;
    let shred_ns = time_fn(
        || shredded = Some(shred_variant(&unshredded, &shape).unwrap()),
        rows,
    );
    let shredded = shredded.unwrap();
    eprintln!("encode (json_to_variant, Rust): {enc_ns:.0} ns/row");
    eprintln!(
        "shred (shred_variant, 6 paths):  {shred_ns:.0} ns/row  (D-14 flush-time cost, Rust side)"
    );

    let text_schema: SchemaRef =
        Arc::new(Schema::new(vec![Field::new("t", DataType::Utf8, false)]));
    let un_field = unshredded.field("v");
    let sh_field = shredded.field("v");
    let sh_schema: SchemaRef = Arc::new(Schema::new(vec![sh_field]));
    // json_to_variant emits BinaryView children; a parquet read yields Binary. Cast to Binary so
    // the layout matches what the engine sees (variant_to_json at 58.3.0 only accepts Binary).
    let un_arr: ArrayRef = {
        let raw: ArrayRef = Arc::new(unshredded.clone().into_inner());
        let target = DataType::Struct(
            vec![
                Arc::new(Field::new("metadata", DataType::Binary, false)),
                Arc::new(Field::new("value", DataType::Binary, true)),
            ]
            .into(),
        );
        datafusion::arrow::compute::cast(&raw, &target).unwrap()
    };
    let un_schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "v",
        un_arr.data_type().clone(),
        false,
    )
    .with_metadata(un_field.metadata().clone())]));
    let sh_arr: ArrayRef = Arc::new(shredded.clone().into_inner());
    eprintln!(
        "in-memory size: text {:.1} MB, unshredded variant {:.1} MB, shredded variant {:.1} MB",
        text.get_array_memory_size() as f64 / 1e6,
        un_arr.get_array_memory_size() as f64 / 1e6,
        sh_arr.get_array_memory_size() as f64 / 1e6
    );

    let tb = batches(&text_schema, &text);
    let ub = batches(&un_schema, &un_arr);
    let sb = batches(&sh_schema, &sh_arr);

    eprintln!(
        "\n{:<48} {:>14} {:>14} {:>14}",
        "query shape", "text+json_extract", "unshredded", "shredded"
    );
    let mut row =
        |label: &str, t: Option<(f64, usize)>, u: Option<(f64, usize)>, s: Option<(f64, usize)>| {
            let f = |x: Option<(f64, usize)>| {
                x.map(|(ns, nn)| {
                    if ns < 1.0 {
                        format!("      <1 ns ({nn})")
                    } else {
                        format!("{ns:>8.0} ns ({nn})")
                    }
                })
                .unwrap_or_else(|| "-".to_string())
            };
            eprintln!("{:<48} {:>14} {:>14} {:>14}", label, f(t), f(u), f(s));
        };

    // 1. field extraction: shredded path (int)
    row(
        "extract responseElements.statusCode (shredded path)",
        Some(time_expr(
            &json_extract(&text_schema, "responseElements.statusCode"),
            &tb,
            rows,
        )),
        Some(time_expr(
            &variant_get(&un_schema, "responseElements.statusCode", Some("Int64")),
            &ub,
            rows,
        )),
        Some(time_expr(
            &variant_get(&sh_schema, "responseElements.statusCode", Some("Int64")),
            &sb,
            rows,
        )),
    );
    // 1b. field extraction: deep path NOT in the shredding schema (path-in-value cost)
    row(
        "extract userIdentity.sessionContext.attributes.mfaAuthenticated",
        Some(time_expr(
            &json_extract(
                &text_schema,
                "userIdentity.sessionContext.attributes.mfaAuthenticated",
            ),
            &tb,
            rows,
        )),
        Some(time_expr(
            &variant_get(
                &un_schema,
                "userIdentity.sessionContext.attributes.mfaAuthenticated",
                Some("Utf8"),
            ),
            &ub,
            rows,
        )),
        Some(time_expr(
            &variant_get(
                &sh_schema,
                "userIdentity.sessionContext.attributes.mfaAuthenticated",
                Some("Utf8"),
            ),
            &sb,
            rows,
        )),
    );
    // 2. full object (reconstruction): text is a passthrough; variant must re-serialise.
    let un_to_json = time_fn(
        || {
            for b in &ub {
                let _ = variant_to_json(b.column(0)).unwrap();
            }
        },
        rows,
    );
    let sh_to_json = time_fn(
        || {
            for b in &sb {
                let _ = crate::udf::variant_get::variant_to_json_text(b.column(0)).unwrap();
            }
        },
        rows,
    );
    eprintln!(
        "{:<48} {:>14} {:>14} {:>14}",
        "full object (SELECT v) reconstruction",
        "0 ns (passthrough)",
        format!("{un_to_json:>8.0} ns"),
        format!("{sh_to_json:>8.0} ns (unshred+json)")
    );
    // 3. array element
    row(
        "array element requestParameters.tags[0].value",
        Some(time_expr(
            &json_extract(&text_schema, "requestParameters.tags[0].value"),
            &tb,
            rows,
        )),
        Some(time_expr(
            &variant_get(&un_schema, "requestParameters.tags[0].value", Some("Utf8")),
            &ub,
            rows,
        )),
        Some(time_expr(
            &variant_get(&sh_schema, "requestParameters.tags[0].value", Some("Utf8")),
            &sb,
            rows,
        )),
    );
    // 4. full array (returned as JSON text in both layouts)
    row(
        "full array resources (as JSON text)",
        Some(time_expr(
            &json_extract(&text_schema, "resources"),
            &tb,
            rows,
        )),
        Some(time_expr(
            &variant_get(&un_schema, "resources", None),
            &ub,
            rows,
        )),
        None,
    );
    eprintln!();
}
