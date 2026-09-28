/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! POC-2 step 5b (D-18 / V-25 / V-27): one `IndexedTable` over segments whose Variant column
//! `v` has *different* physical layouts.
//!
//! Segments (all `id: Int32, v: Variant`, 4096 rows, `v.x` laid out like the page-pruning
//! fixture: page p holds `p*10_000 .. p*10_000+1024`):
//!
//! * **P1** unshredded: `v = Struct<metadata, value>`, 1 RG, 4 pages of 1024. ids 0..4096.
//! * **P2** shredded on `x: Int64` (the adversarial fixture from `variant_pruning.rs`: 4 RGs of
//!   1024, pages of 256, one `{"x":"200"}` string row in RG 2 page 1). ids 4096..8192.
//! * **P2'** shredded on `y: Int64` only, docs `{"x": n, "y": n}` so `x` lives in the residual
//!   `value` while `typed_value` exists. ids 8192..12288.
//! * **P2''** shredded on `x: Utf8` — a declared type change (D-20). Only used to show where the
//!   union schema breaks.
//!
//! Segments and the table schema come from the production `build_segments` (DataFusion
//! `ParquetFormat::infer_schema` union), not the footer-verbatim path the other Variant tests
//! use, so this file also measures V-27 on the real IndexedTable path.
//!
//! What it found out of the box (2026-09-28, DF 54 / arrow-rs 58.3.0), fixed by D-18(a):
//!
//! 1. `infer_schema` accepts the mixed footers (nested `Field::try_merge` by name) but strips the
//!    `arrow.parquet.variant` tag (`skip_metadata = true`), so `variant_get` refused `v` at plan
//!    time — V-27 hits the production IndexedTable path too.
//! 2. With the tag restored, the union type `Struct<metadata, value, typed_value<x, y>>` is read
//!    through DataFusion's name-based nested-struct cast, which fills a segment's absent
//!    `typed_value` with NULLs. arrow-rs `variant_get` then returns NULL for every such row
//!    instead of reading `value` (`kernels_on_p1_array_adapted_to_p2_layout`): the 2-file table
//!    silently lost all 1024 P1 matches. Three files fail earlier: the union keeps shredded field
//!    structs non-nullable, so `Cannot cast struct: target field 'y' is non-nullable but missing
//!    from source` at scan time.
//! 3. A declared type change (`x: Int64` beside `x: Utf8`) does not merge at all:
//!    `Fail to merge schema field 'typed_value'` (D-20's v1 ban is what keeps this out).
//!
//! D-18(a) as implemented: `build_segments` exposes every Variant column as the canonical
//! `Struct<metadata, value>` (tagged), each segment is read against its own footer, and
//! `variant_adapter.rs` replaces DataFusion's struct cast with `unshred_variant` for segments
//! whose layout differs. Pruning was already per segment (`eval_leaf` / `PagePruner` resolve
//! leaves against `seg_arrow_schema`; absent leaves are conservative).
//!
//! Per query the test compares the table's answer with a per-segment brute force (same UDF on
//! the in-memory `VariantArray`) and reports per-segment RG / page pruning counters.
//!
//! All tests are `#[ignore]` release e2e tests:
//! `cargo test --release -p opensearch-datafusion variant_mixed_schema -- --ignored --nocapture`

#![cfg(test)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::ScalarValue;
use datafusion::config::ConfigOptions;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Operator, ScalarUDF};
use datafusion::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::{EnabledStatistics, WriterProperties};
use datafusion::physical_expr::expressions::{
    is_not_null, BinaryExpr, Column as PhysColumn, Literal,
};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_plan::metrics::Count;
use parquet::variant::{json_to_variant, shred_variant, ShreddedSchemaBuilder, VariantArray};
use tempfile::NamedTempFile;

use crate::indexed_table::bool_tree::BoolNode;
use crate::indexed_table::eval::bitmap_tree::{BitmapTreeEvaluator, CollectorLeafBitmaps};
use crate::indexed_table::eval::{CollectorCallStrategy, RowGroupBitsetSource, TreeBitsetSource};
use crate::indexed_table::index::{CollectDocsResult, RowGroupDocsCollector};
use crate::indexed_table::nested_leaf::FlatLeafView;
use crate::indexed_table::page_pruner::{
    build_pruning_predicate, PagePruneMetrics, PagePruner, StatsPruneTree,
};
use crate::indexed_table::segment_info::build_segments;
use crate::indexed_table::stream::FilterStrategy;
use crate::indexed_table::table_provider::{
    EvaluatorFactory, IndexedTableConfig, IndexedTableProvider, SegmentFileInfo,
};
use crate::udf::variant_get::{variant_to_json_text, VariantGetUdf};

const ROWS_PER_PAGE: usize = 1024;
const NUM_PAGES: usize = 4;
const NUM_ROWS: usize = ROWS_PER_PAGE * NUM_PAGES;
const SHREDDED_PAGE_ROWS: usize = 256;
/// Row index of the adversarial string row in P2 (RG 2, page 1).
const FALLBACK_ROW: usize = 2 * ROWS_PER_PAGE + 500;
const EXT_KEY: &str = "ARROW:extension:name";
const TYPED_LEAF: &str = "v.typed_value.x.typed_value";
const VALUE_LEAF: &str = "v.typed_value.x.value";

// ── Fixtures ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// P1: no `typed_value`.
    Unshredded,
    /// P2: `typed_value.x: Int64`, adversarial string row.
    ShreddedX,
    /// P2': `typed_value.y: Int64`, `x` only in `value`.
    ShreddedY,
    /// P2'': `typed_value.x: Utf8` (declared type change).
    ShreddedXUtf8,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Unshredded => "P1(unshredded)",
            Shape::ShreddedX => "P2(x:Int64)",
            Shape::ShreddedY => "P2'(y:Int64)",
            Shape::ShreddedXUtf8 => "P2''(x:Utf8)",
        }
    }
}

fn x_value(row: usize) -> i64 {
    ((row / ROWS_PER_PAGE) as i64) * 10_000 + (row % ROWS_PER_PAGE) as i64
}

fn json_docs(shape: Shape) -> ArrayRef {
    let docs: Vec<String> = (0..NUM_ROWS)
        .map(|r| match shape {
            Shape::ShreddedX if r == FALLBACK_ROW => "{\"x\":\"200\"}".to_string(),
            Shape::ShreddedY => format!("{{\"x\":{},\"y\":{}}}", x_value(r), x_value(r)),
            _ => format!("{{\"x\":{}}}", x_value(r)),
        })
        .collect();
    Arc::new(StringArray::from(docs))
}

fn variant_column(shape: Shape) -> VariantArray {
    let unshredded = json_to_variant(&json_docs(shape)).unwrap();
    let shred = match shape {
        Shape::Unshredded => return unshredded,
        Shape::ShreddedX => ShreddedSchemaBuilder::new().with_path("x", (&DataType::Int64, true)),
        Shape::ShreddedY => ShreddedSchemaBuilder::new().with_path("y", (&DataType::Int64, true)),
        Shape::ShreddedXUtf8 => {
            ShreddedSchemaBuilder::new().with_path("x", (&DataType::Utf8, true))
        }
    };
    shred_variant(&unshredded, &shred.unwrap().build()).unwrap()
}

/// In-memory `[id, v]` batch for one segment; also the brute-force oracle.
fn in_memory_batch(shape: Shape, id_base: i32) -> RecordBatch {
    let variant = variant_column(shape);
    let v_field = variant.field("v");
    let v: ArrayRef = Arc::new(variant.into_inner());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        v_field,
    ]));
    let ids: ArrayRef = Arc::new(Int32Array::from(
        (id_base..id_base + NUM_ROWS as i32).collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema, vec![ids, v]).unwrap()
}

fn write_fixture(shape: Shape, id_base: i32) -> NamedTempFile {
    let batch = in_memory_batch(shape, id_base);
    let (rg_rows, page_rows) = match shape {
        Shape::Unshredded => (NUM_ROWS, ROWS_PER_PAGE),
        _ => (ROWS_PER_PAGE, SHREDDED_PAGE_ROWS),
    };
    let props = WriterProperties::builder()
        .set_max_row_group_size(rg_rows)
        .set_data_page_row_count_limit(page_rows)
        .set_write_batch_size(page_rows)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let tmp = NamedTempFile::new().unwrap();
    let mut w = ArrowWriter::try_new(tmp.reopen().unwrap(), batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    tmp
}

fn object_meta(tmp: &NamedTempFile) -> object_store::ObjectMeta {
    let path = tmp.path();
    object_store::ObjectMeta {
        location: object_store::path::Path::from(path.to_string_lossy().as_ref()),
        last_modified: chrono::Utc::now(),
        size: std::fs::metadata(path).unwrap().len(),
        e_tag: None,
        version: None,
    }
}

/// Production path: `build_segments` → per-segment footer schema + DataFusion union schema.
///
/// The footer metadata that `build_segments` caches is loaded with `PageIndexPolicy::Skip`
/// (the scoped page-index cache fetches it lazily per column in production). This harness
/// bypasses that cache, so re-load each segment's metadata with the page index attached —
/// same as `load_segment` in `variant_pruning.rs` — to keep page-level counters observable.
async fn build_table(
    ctx: &SessionContext,
    tmps: &[NamedTempFile],
) -> Result<(Vec<SegmentFileInfo>, SchemaRef), String> {
    let store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let metas: Vec<_> = tmps.iter().map(object_meta).collect();
    let gens: Vec<i64> = (0..tmps.len() as i64).collect();
    let cache = ctx
        .state()
        .runtime_env()
        .cache_manager
        .get_file_metadata_cache();
    let (mut segments, schema) =
        build_segments(&ctx.state(), store, &metas, &gens, cache, &[]).await?;
    for (seg, tmp) in segments.iter_mut().zip(tmps) {
        let file = std::fs::File::open(tmp.path()).unwrap();
        let meta =
            ArrowReaderMetadata::load(&file, ArrowReaderOptions::new().with_page_index(true))
                .unwrap();
        seg.metadata = meta.metadata().clone();
    }
    Ok((segments, schema))
}

// ── Expressions ─────────────────────────────────────────────────────

fn col(schema: &Schema, name: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(PhysColumn::new(name, schema.index_of(name).unwrap()))
}

fn lit_str(v: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Utf8(Some(v.to_string()))))
}

fn lit_i64(v: i64) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int64(Some(v))))
}

fn binop(
    l: Arc<dyn PhysicalExpr>,
    op: Operator,
    r: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(l, op, r))
}

/// `variant_get(v, path [, ty])` bound to `schema`.
fn variant_get(schema: &Schema, path: &str, ty: Option<&str>) -> Arc<dyn PhysicalExpr> {
    let udf = Arc::new(ScalarUDF::from(VariantGetUdf::new()));
    let mut args = vec![col(schema, "v"), lit_str(path)];
    if let Some(ty) = ty {
        args.push(lit_str(ty));
    }
    Arc::new(
        ScalarFunctionExpr::try_new(udf, args, schema, Arc::new(ConfigOptions::default())).unwrap(),
    )
}

/// `variant_get(v,'$.x','Int64') < 1024`.
fn udf_predicate(schema: &Schema) -> Arc<dyn PhysicalExpr> {
    binop(
        variant_get(schema, "$.x", Some("Int64")),
        Operator::Lt,
        lit_i64(1024),
    )
}

/// D-08 guard on the flat leaves: `typed < 1024 OR value IS NOT NULL`.
fn guarded_predicate(pruning_schema: &Schema) -> Arc<dyn PhysicalExpr> {
    let naive = binop(col(pruning_schema, TYPED_LEAF), Operator::Lt, lit_i64(1024));
    Arc::new(BinaryExpr::new(
        naive,
        Operator::Or,
        is_not_null(col(pruning_schema, VALUE_LEAF)).unwrap(),
    ))
}

/// Table schema + every nested parquet leaf seen in any segment (dotted names), so a pruning
/// predicate on `v.typed_value.x.*` can be built once at table level while each segment resolves
/// (or fails to resolve, conservatively) the leaves against its own footer.
fn pruning_schema(table: &Schema, segments: &[SegmentFileInfo]) -> SchemaRef {
    let mut fields: Vec<Field> = table.fields().iter().map(|f| f.as_ref().clone()).collect();
    for seg in segments {
        if let Some(view) = FlatLeafView::build(
            &seg.arrow_schema,
            seg.metadata.file_metadata().schema_descr(),
        ) {
            for f in view.schema().fields() {
                if fields.iter().all(|x| x.name() != f.name()) {
                    fields.push(f.as_ref().clone());
                }
            }
        }
    }
    Arc::new(Schema::new(fields))
}

// ── Harness ─────────────────────────────────────────────────────────

#[derive(Debug)]
struct AllDocs;

impl RowGroupDocsCollector for AllDocs {
    fn collect_packed_u64_bitset(
        &self,
        min_doc: i32,
        max_doc: i32,
    ) -> Result<CollectDocsResult, String> {
        let span = (max_doc - min_doc) as usize;
        let mut out = vec![0u64; span.div_ceil(64)];
        for rel in 0..span {
            out[rel / 64] |= 1u64 << (rel % 64);
        }
        Ok(out.into())
    }
}

/// Per-segment pruning report (writer_generation → counters).
#[derive(Default, Clone, Debug)]
struct SegReport {
    rg_keep: Vec<bool>,
    pages_pruned: Count,
    pages_total: Count,
    page_pruning_unavailable: Count,
}

struct RunResult {
    batches: Vec<RecordBatch>,
    reports: Vec<SegReport>,
}

/// Runs `AND(all-docs collector, [predicate])` through `IndexedTableProvider` over `segments`
/// and returns the SQL result plus per-segment pruning counters.
///
/// `pp_override` replaces the pruning predicate attached to the UDF leaf (the D-09 "stats half"
/// rewrite: prune with the guarded flat-leaf predicate, refine with the real UDF).
async fn run(
    ctx: &SessionContext,
    segments: Vec<SegmentFileInfo>,
    schema: SchemaRef,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    pp_override: Option<Arc<dyn PhysicalExpr>>,
    sql: &str,
) -> Result<RunResult, String> {
    let prune_schema = pruning_schema(&schema, &segments);
    let has_predicate = predicate.is_some();
    let mut children = vec![BoolNode::Collector { annotation_id: 0 }];
    let mut pp_map: HashMap<usize, Arc<datafusion::physical_optimizer::pruning::PruningPredicate>> =
        HashMap::new();
    if let Some(p) = predicate {
        let key = Arc::as_ptr(&p) as *const () as usize;
        let pp = match &pp_override {
            Some(g) => build_pruning_predicate(g, prune_schema.clone()),
            None => build_pruning_predicate(&p, prune_schema.clone()),
        };
        if let Some(pp) = pp {
            pp_map.insert(key, pp);
        }
        children.push(BoolNode::Predicate(p));
    }
    let tree = Arc::new(BoolNode::And(children).push_not_down());
    let pp_map = Arc::new(pp_map);

    // RG-level decisions per segment, exactly as table_provider.rs:668 builds them.
    let mut reports: Vec<SegReport> = Vec::with_capacity(segments.len());
    for seg in &segments {
        let rg_indices: Vec<usize> = seg.row_groups.iter().map(|rg| rg.index).collect();
        let spt = StatsPruneTree::build_from_bool_node(
            &tree,
            &pp_map,
            &seg.metadata,
            &prune_schema,
            &rg_indices,
            &seg.arrow_schema,
        );
        reports.push(SegReport {
            rg_keep: spt.rg_can_match.clone(),
            ..Default::default()
        });
    }
    let counters: Arc<Mutex<HashMap<i64, PagePruneMetrics>>> = Arc::new(Mutex::new(
        segments
            .iter()
            .zip(&reports)
            .map(|(s, r)| {
                (
                    s.writer_generation,
                    PagePruneMetrics {
                        pages_pruned: Some(r.pages_pruned.clone()),
                        pages_total: Some(r.pages_total.clone()),
                        page_pruning_unavailable: Some(r.page_pruning_unavailable.clone()),
                    },
                )
            })
            .collect(),
    ));

    let per_leaf: Vec<(i32, Arc<dyn RowGroupDocsCollector>)> = vec![(0, Arc::new(AllDocs))];
    let factory: EvaluatorFactory = {
        let tree = Arc::clone(&tree);
        let prune_schema = prune_schema.clone();
        let pp_map = Arc::clone(&pp_map);
        let counters = Arc::clone(&counters);
        Arc::new(move |segment, chunk, sm, spt| {
            let resolved = tree.resolve(&per_leaf)?;
            let pruner = Arc::new(PagePruner::new(
                &prune_schema,
                Arc::clone(&segment.metadata),
                segment.arrow_schema.clone(),
            ));
            let metrics = counters
                .lock()
                .unwrap()
                .get(&segment.writer_generation)
                .cloned()
                .unwrap_or_default();
            let eval: Arc<dyn RowGroupBitsetSource> = Arc::new(TreeBitsetSource {
                tree: Arc::new(resolved),
                evaluator: Arc::new(BitmapTreeEvaluator),
                leaves: Arc::new(CollectorLeafBitmaps::new(sm.ffm_collector_calls.clone())),
                page_pruner: pruner,
                cost_predicate: 1,
                cost_collector: 10,
                max_collector_parallelism: 1,
                pruning_predicates: Arc::clone(&pp_map),
                page_prune_metrics: Some(metrics),
                collector_strategy: CollectorCallStrategy::TightenOuterBounds,
                stats_prune_tree: spt.cloned(),
                rg_index_to_pos: chunk
                    .row_group_indices
                    .iter()
                    .enumerate()
                    .map(|(pos, &idx)| (idx, pos))
                    .collect(),
            });
            Ok(eval)
        })
    };

    let store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let store_url = datafusion::execution::object_store::ObjectStoreUrl::local_filesystem();
    let qc = crate::datafusion_query_config::DatafusionQueryConfig::builder()
        .target_partitions(1)
        .force_strategy(Some(FilterStrategy::BooleanMask))
        .indexed_pushdown_filters(false)
        .build();
    let predicate_columns = if has_predicate {
        vec![schema.index_of("v").unwrap()]
    } else {
        vec![]
    };
    let provider = Arc::new(IndexedTableProvider::new(IndexedTableConfig {
        schema: schema.clone(),
        segments,
        store,
        store_url,
        evaluator_factory: factory,
        pushdown_predicate: None,
        query_config: Arc::new(qc),
        predicate_columns,
        emit_row_ids: false,
        prune_tree_config: Some((Arc::clone(&tree), Arc::clone(&pp_map), prune_schema)),
        sort_fields: vec![],
        sort_orders: vec![],
        cancellation_token: None,
    }));
    let _ = ctx.deregister_table("t");
    ctx.register_table("t", provider)
        .map_err(|e| e.to_string())?;
    let df = ctx
        .sql(sql)
        .await
        .map_err(|e| format!("plan `{sql}`: {e}"))?;
    let batches = df
        .collect()
        .await
        .map_err(|e| format!("execute `{sql}`: {e}"))?;
    Ok(RunResult { batches, reports })
}

fn fmt_report(labels: &[Shape], reports: &[SegReport]) -> String {
    labels
        .iter()
        .zip(reports)
        .map(|(l, r)| {
            let kept = r.rg_keep.iter().filter(|k| **k).count();
            format!(
                "{}: RGs kept {}/{} {:?}, pages pruned {}/{} (unavailable={})",
                l.label(),
                kept,
                r.rg_keep.len(),
                r.rg_keep,
                r.pages_pruned.value(),
                r.pages_total.value(),
                r.page_pruning_unavailable.value()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

// ── Oracles ─────────────────────────────────────────────────────────

fn i32_col(batches: &[RecordBatch], idx: usize) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .map(|v| v.unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn str_col(batches: &[RecordBatch], idx: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|b| {
            let arr = b.column(idx);
            let arr = if arr.data_type() == &DataType::Utf8 {
                Arc::clone(arr)
            } else {
                datafusion::arrow::compute::cast(arr, &DataType::Utf8).unwrap()
            };
            arr.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|v| v.map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Brute force for `variant_get(v,'$.x','Int64') < 1024` on one in-memory segment.
fn brute_ids_lt_1024(batch: &RecordBatch) -> Vec<i32> {
    let extracted = variant_get(&batch.schema(), "$.x", Some("Int64"))
        .evaluate(batch)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap();
    let extracted = extracted.as_any().downcast_ref::<Int64Array>().unwrap();
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..batch.num_rows())
        .filter(|r| !extracted.is_null(*r) && extracted.value(*r) < 1024)
        .map(|r| ids.value(r))
        .collect()
}

/// Brute force for `variant_get(v,'$.x')` (JSON text) on one in-memory segment.
fn brute_x_json(batch: &RecordBatch) -> Vec<(i32, Option<String>)> {
    let extracted = variant_get(&batch.schema(), "$.x", None)
        .evaluate(batch)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap();
    let extracted = extracted.as_any().downcast_ref::<StringArray>().unwrap();
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..batch.num_rows())
        .map(|r| {
            let v = if extracted.is_null(r) {
                None
            } else {
                Some(extracted.value(r).to_string())
            };
            (ids.value(r), v)
        })
        .collect()
}

/// Brute force for `SELECT v` rendered as JSON text on one in-memory segment.
fn brute_v_json(batch: &RecordBatch) -> Vec<(i32, Option<String>)> {
    let json = variant_to_json_text(batch.column(1)).unwrap();
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..batch.num_rows())
        .map(|r| {
            let v = if json.is_null(r) {
                None
            } else {
                Some(json.value(r).to_string())
            };
            (ids.value(r), v)
        })
        .collect()
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

// ── The mixed-table scenario, parameterised over the segment set ────

async fn mixed_table_scenario(shapes: &[Shape]) {
    let ctx = SessionContext::new();
    crate::udf::variant_get::register_all(&ctx);
    let tmps: Vec<NamedTempFile> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| write_fixture(*s, (i * NUM_ROWS) as i32))
        .collect();
    let oracles: Vec<RecordBatch> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| in_memory_batch(*s, (i * NUM_ROWS) as i32))
        .collect();

    // ── Schema plumbing: does the production union accept the footers? ──
    let (segments, schema) = build_table(&ctx, &tmps)
        .await
        .expect("build_segments must accept differently shredded Variant footers");
    let v = schema.field_with_name("v").unwrap();
    eprintln!("[{}] union v = {v:#?}", shapes.len());
    for (s, seg) in shapes.iter().zip(&segments) {
        eprintln!(
            "  footer {} v = {}",
            s.label(),
            seg.arrow_schema.field_with_name("v").unwrap()
        );
    }
    // V-27 on the IndexedTable production path: the tag must survive the union. D-18(a): the
    // table type is the canonical unshredded struct whatever the segments' layouts.
    assert!(
        v.metadata().contains_key(EXT_KEY),
        "V-27: `build_segments` union schema lost the extension tag on `v`: {v:?}"
    );
    assert_eq!(v.metadata()[EXT_KEY], "arrow.parquet.variant");
    let canonical = DataType::Struct(
        vec![
            Arc::new(Field::new("metadata", DataType::Binary, false)),
            Arc::new(Field::new("value", DataType::Binary, true)),
        ]
        .into(),
    );
    assert_eq!(
        v.data_type(),
        &canonical,
        "table `v` must be the canonical Variant struct"
    );

    let n_files = shapes.len();

    // ── (1) UDF predicate, opaque pruning ──
    let expected_lt: Vec<i32> = sorted(oracles.iter().flat_map(brute_ids_lt_1024).collect());
    let r = run(
        &ctx,
        segments.clone(),
        schema.clone(),
        Some(udf_predicate(&schema)),
        None,
        "SELECT id FROM t",
    )
    .await
    .unwrap();
    let got = sorted(i32_col(&r.batches, 0));
    let per_seg = |ids: &[i32]| -> Vec<usize> {
        (0..n_files)
            .map(|i| {
                let lo = (i * NUM_ROWS) as i32;
                ids.iter()
                    .filter(|id| **id >= lo && **id < lo + NUM_ROWS as i32)
                    .count()
            })
            .collect()
    };
    eprintln!(
        "[{n_files}-file] (1) variant_get(v,'$.x','Int64') < 1024 -> {} rows per segment {:?} (expected {:?}); {}",
        got.len(),
        per_seg(&got),
        per_seg(&expected_lt),
        fmt_report(shapes, &r.reports)
    );
    assert_eq!(
        got, expected_lt,
        "(1) mixed table != per-segment brute force"
    );
    for (s, rep) in shapes.iter().zip(&r.reports) {
        assert!(
            rep.rg_keep.iter().all(|k| *k),
            "(1) opaque UDF must not prune RGs on {}: {:?}",
            s.label(),
            rep.rg_keep
        );
        assert_eq!(
            rep.pages_pruned.value(),
            0,
            "(1) opaque UDF pruned pages on {}",
            s.label()
        );
    }

    // (1b) the same predicate under COUNT(*) — exercises the empty read projection.
    let r = run(
        &ctx,
        segments.clone(),
        schema.clone(),
        Some(udf_predicate(&schema)),
        None,
        "SELECT count(*) FROM t",
    )
    .await
    .unwrap();
    let cnt = r.batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    eprintln!("[{n_files}-file] (1b) count(*) -> {cnt}");
    assert_eq!(cnt as usize, expected_lt.len());

    // ── (2) UDF predicate refined, guarded flat-leaf predicate for pruning ──
    let prune_schema = pruning_schema(&schema, &segments);
    let r = run(
        &ctx,
        segments.clone(),
        schema.clone(),
        Some(udf_predicate(&schema)),
        Some(guarded_predicate(&prune_schema)),
        "SELECT id FROM t",
    )
    .await
    .unwrap();
    let got = sorted(i32_col(&r.batches, 0));
    eprintln!(
        "[{n_files}-file] (2) same, pruned by `typed<1024 OR value IS NOT NULL` -> {} rows; {}",
        got.len(),
        fmt_report(shapes, &r.reports)
    );
    assert_eq!(got, expected_lt, "(2) guarded pruning changed the answer");
    for (s, rep) in shapes.iter().zip(&r.reports) {
        match s {
            Shape::ShreddedX => {
                assert_eq!(
                    rep.rg_keep,
                    vec![true, false, true, false],
                    "P2 must still prune RGs 1 and 3 inside the mixed table"
                );
                // RGs 0 and 2 survive (8 pages); in RG 2 only the fallback row's page is kept.
                assert_eq!(rep.pages_total.value(), 8, "P2 pages considered");
                assert_eq!(
                    rep.pages_pruned.value(),
                    3,
                    "P2 must still prune the 3 non-fallback pages of RG 2 inside the mixed table"
                );
            }
            _ => {
                assert!(
                    rep.rg_keep.iter().all(|k| *k),
                    "{} has no typed leaf for x and must not prune: {:?}",
                    s.label(),
                    rep.rg_keep
                );
                assert_eq!(rep.pages_pruned.value(), 0, "{} pruned pages", s.label());
            }
        }
    }

    // ── (3) untyped extraction as JSON text, all rows ──
    let expected_json: Vec<(i32, Option<String>)> =
        sorted(oracles.iter().flat_map(brute_x_json).collect());
    let r = run(
        &ctx,
        segments.clone(),
        schema.clone(),
        None,
        None,
        "SELECT id, variant_get(v, '$.x') FROM t",
    )
    .await
    .unwrap();
    let got: Vec<(i32, Option<String>)> = sorted(
        i32_col(&r.batches, 0)
            .into_iter()
            .zip(str_col(&r.batches, 1))
            .collect(),
    );
    eprintln!(
        "[{n_files}-file] (3) SELECT variant_get(v,'$.x') -> {} rows, sample {:?} .. {:?}",
        got.len(),
        &got[..2],
        &got[got.len() - 2..]
    );
    assert_eq!(got.len(), n_files * NUM_ROWS);
    assert_eq!(
        got, expected_json,
        "(3) untyped extraction differs from brute force"
    );

    // ── (4) reconstruction: SELECT v ──
    let expected_v: Vec<(i32, Option<String>)> =
        sorted(oracles.iter().flat_map(brute_v_json).collect());
    let r = run(
        &ctx,
        segments.clone(),
        schema.clone(),
        None,
        None,
        "SELECT id, v FROM t",
    )
    .await
    .unwrap();
    let out_v_type = r.batches[0].column(1).data_type().clone();
    let mut got: Vec<(i32, Option<String>)> = Vec::new();
    for b in &r.batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let json = variant_to_json_text(b.column(1)).expect("output `v` must be a valid Variant");
        for i in 0..b.num_rows() {
            let v = if json.is_null(i) {
                None
            } else {
                Some(json.value(i).to_string())
            };
            got.push((ids.value(i), v));
        }
    }
    let got = sorted(got);
    eprintln!(
        "[{n_files}-file] (4) SELECT v -> {} rows as {out_v_type}, sample {:?} / {:?}",
        got.len(),
        &got[..1],
        &got[FALLBACK_ROW + NUM_ROWS..FALLBACK_ROW + NUM_ROWS + 1]
    );
    assert_eq!(
        got, expected_v,
        "(4) reconstructed `v` differs from brute force"
    );
}

// ═════════════════════════════════════════════════════════════════════

/// Two files: P1 unshredded + P2 shredded on `x`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn two_file_table_unshredded_plus_shredded_x() {
    mixed_table_scenario(&[Shape::Unshredded, Shape::ShreddedX]).await;
}

/// Three files: P1 + P2 + P2' shredded on a different path (`y`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn three_file_table_adds_shredded_y() {
    mixed_table_scenario(&[Shape::Unshredded, Shape::ShreddedX, Shape::ShreddedY]).await;
}

/// Why option D-18(b) (union schema, absent `typed_value` filled with NULL) is unsound with
/// arrow-rs 58.3.0: adapt an unshredded (P1) array to the shredded (P2) layout exactly the way
/// DataFusion's nested-struct cast does it (`typed_value` = all-NULL struct). `unshred_variant`
/// reads such rows from `value` (spec table: value non-NULL, typed_value NULL = "any type"),
/// but `variant_get` unions `typed_value`'s nulls into the result (`variant_get.rs:163`) and
/// never consults `value`, so every P1 row becomes NULL. If this test starts failing, the kernel
/// gained the row-level fallback and (b) can be reconsidered.
#[test]
#[ignore]
fn kernels_on_p1_array_adapted_to_p2_layout() {
    let p1 = in_memory_batch(Shape::Unshredded, 0);
    let p2 = in_memory_batch(Shape::ShreddedX, 0);
    let target = p2.column(1).data_type().clone();
    let adapted = datafusion::common::nested_struct::cast_column(
        p1.column(1),
        &target,
        &datafusion::arrow::compute::CastOptions::default(),
    )
    .expect("DF nested-struct cast P1 -> P2 layout");
    let adapted_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            p2.schema().field(1).clone(),
        ])),
        vec![Arc::clone(p1.column(0)), adapted],
    )
    .unwrap();
    let typed = variant_get(&adapted_batch.schema(), "$.x", Some("Int64"))
        .evaluate(&adapted_batch)
        .unwrap()
        .into_array(NUM_ROWS)
        .unwrap();
    let untyped = variant_get(&adapted_batch.schema(), "$.x", None)
        .evaluate(&adapted_batch)
        .unwrap()
        .into_array(NUM_ROWS)
        .unwrap();
    let json = variant_to_json_text(adapted_batch.column(1)).unwrap();
    eprintln!(
        "[kernels] P1 adapted to P2 layout: typed nulls={} first={:?}; untyped nulls={} first={:?}; to_json nulls={} first={:?}",
        typed.null_count(),
        ScalarValue::try_from_array(&typed, 0).unwrap(),
        untyped.null_count(),
        ScalarValue::try_from_array(&untyped, 0).unwrap(),
        json.null_count(),
        json.value(0)
    );
    assert_eq!(
        typed.null_count(),
        NUM_ROWS,
        "arrow-rs variant_get now falls back to `value` behind a NULL typed_value — re-evaluate D-18(b)"
    );
    assert_eq!(untyped.null_count(), NUM_ROWS);
    assert_eq!(
        json.null_count(),
        0,
        "unshred_variant must read the rows from `value`"
    );
}

/// D-20 type change: `x: Int64` beside `x: Utf8`. The union schema cannot represent it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn shredded_x_int64_beside_x_utf8_fails_at_schema_merge() {
    let ctx = SessionContext::new();
    let tmps = vec![
        write_fixture(Shape::ShreddedX, 0),
        write_fixture(Shape::ShreddedXUtf8, NUM_ROWS as i32),
    ];
    let err = build_table(&ctx, &tmps)
        .await
        .err()
        .expect("Int64 vs Utf8 typed leaf for the same path must not merge");
    eprintln!("[type-change] build_segments error: {err}");
    assert!(
        err.contains("infer_schema union") && err.contains("merge"),
        "expected a schema-merge failure, got: {err}"
    );
}
