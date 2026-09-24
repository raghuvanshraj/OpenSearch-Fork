/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! POC-2: Variant read path against the indexed table's three pruning levels.
//!
//! Two fixtures, both `id: Int32, v: Variant` with `v = {"x": <n>}` and `n` laid out like the
//! page-pruning fixture (page p holds `p*10_000 .. p*10_000+1024`), 4096 rows:
//!
//! * **unshredded**: 1 RG, 4 pages of 1024. `v` is `Struct<metadata, value>`.
//! * **shredded (adversarial)**: 4 RGs of 1024, 4 pages of 256 each, `v` shredded with
//!   `typed_value: Struct<x: Struct<value, typed_value: Int64>>`. Exactly one row, id 2548
//!   (RG 2, page 1), holds `{"x": "200"}` — a *string*, so it falls back to `x.value` and
//!   `x.typed_value` is NULL there. Typed stats for RG 2 are `[20000, 21023]`, which a naive
//!   rewrite of `variant_get(v,'x','Int64') < 1024` would prune even though the fallback row
//!   may satisfy the predicate under lenient casting.
//!
//! What is proven:
//! 1. `variant_get` lands in a projection directly above the scan without any rule of ours
//!    (`placement() = MoveTowardsLeafNodes` + DataFusion's `ExtractLeafExpressions`).
//! 2. An opaque `variant_get` predicate prunes nothing at RG or page level and returns exact
//!    results, alone, under NOT, in OR, and in AND with a prunable typed predicate (V-24).
//! 3. Nested-leaf stats are reachable through [`FlatLeafView`] (V-17 was falsified for the
//!    stock `StatisticsConverter`; this is the extension).
//! 4. On the adversarial file the naive typed-only predicate prunes RG 2 (the hazard) while the
//!    OR-shaped guard `typed < c OR value IS NOT NULL` keeps exactly RGs 0 and 2 and, inside
//!    RG 2, exactly the page holding the fallback row (D-08).

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::config::ConfigOptions;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{LogicalPlan, Operator, ScalarUDF};
use datafusion::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::{EnabledStatistics, WriterProperties};
use datafusion::physical_expr::expressions::{
    is_not_null, BinaryExpr, Column as PhysColumn, Literal,
};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::ParquetReadOptions;
use futures::StreamExt;
use parquet::variant::{json_to_variant, shred_variant, ShreddedSchemaBuilder, VariantArray};
use tempfile::NamedTempFile;

use crate::indexed_table::bool_tree::BoolNode;
use crate::indexed_table::eval::bitmap_tree::{BitmapTreeEvaluator, CollectorLeafBitmaps};
use crate::indexed_table::eval::{CollectorCallStrategy, RowGroupBitsetSource, TreeBitsetSource};
use crate::indexed_table::index::{CollectDocsResult, RowGroupDocsCollector};
use crate::indexed_table::nested_leaf::FlatLeafView;
use crate::indexed_table::page_pruner::{build_pruning_predicate, PagePruner, StatsPruneTree};
use crate::indexed_table::stream::{FilterStrategy, RowGroupInfo};
use crate::indexed_table::table_provider::{
    EvaluatorFactory, IndexedTableConfig, IndexedTableProvider, SegmentFileInfo,
};
use crate::udf::variant_get::VariantGetUdf;

const ROWS_PER_PAGE: usize = 1024;
const NUM_PAGES: usize = 4;
const NUM_ROWS: usize = ROWS_PER_PAGE * NUM_PAGES;
const SHREDDED_PAGE_ROWS: usize = 256;
/// Row index of the adversarial string row in the shredded fixture (RG 2, page 1).
const FALLBACK_ROW: usize = 2 * ROWS_PER_PAGE + 500;

// ── Fixtures ────────────────────────────────────────────────────────

fn x_value(row: usize) -> i64 {
    ((row / ROWS_PER_PAGE) as i64) * 10_000 + (row % ROWS_PER_PAGE) as i64
}

fn json_docs(adversarial: bool) -> ArrayRef {
    let docs: Vec<String> = (0..NUM_ROWS)
        .map(|r| {
            if adversarial && r == FALLBACK_ROW {
                "{\"x\":\"200\"}".to_string()
            } else {
                format!("{{\"x\":{}}}", x_value(r))
            }
        })
        .collect();
    Arc::new(StringArray::from(docs))
}

fn variant_column(shredded: bool) -> VariantArray {
    let unshredded = json_to_variant(&json_docs(shredded)).unwrap();
    if !shredded {
        return unshredded;
    }
    let shape = ShreddedSchemaBuilder::new()
        .with_path("x", (&DataType::Int64, true))
        .unwrap()
        .build();
    shred_variant(&unshredded, &shape).unwrap()
}

fn write_fixture(shredded: bool) -> (NamedTempFile, SchemaRef) {
    let variant = variant_column(shredded);
    let v_field = variant.field("v");
    let v_array: ArrayRef = Arc::new(variant.into_inner());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        v_field,
    ]));
    let ids: ArrayRef = Arc::new(Int32Array::from((0..NUM_ROWS as i32).collect::<Vec<_>>()));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, v_array]).unwrap();
    let (rg_rows, page_rows) = if shredded {
        (ROWS_PER_PAGE, SHREDDED_PAGE_ROWS)
    } else {
        (NUM_ROWS, ROWS_PER_PAGE)
    };
    let props = WriterProperties::builder()
        .set_max_row_group_size(rg_rows)
        .set_data_page_row_count_limit(page_rows)
        .set_write_batch_size(page_rows)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let tmp = NamedTempFile::new().unwrap();
    let mut w = ArrowWriter::try_new(tmp.reopen().unwrap(), schema.clone(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    (tmp, schema)
}

fn load_segment(tmp: &NamedTempFile) -> (SegmentFileInfo, SchemaRef) {
    let path = tmp.path().to_path_buf();
    let size = std::fs::metadata(&path).unwrap().len();
    let file = std::fs::File::open(&path).unwrap();
    let meta =
        ArrowReaderMetadata::load(&file, ArrowReaderOptions::new().with_page_index(true)).unwrap();
    // The footer round trip must re-attach the extension type, or variant_get refuses `v`.
    let schema = meta.schema().clone();
    let parquet_meta = meta.metadata().clone();
    let mut rgs = Vec::new();
    let mut offset = 0i64;
    for i in 0..parquet_meta.num_row_groups() {
        let n = parquet_meta.row_group(i).num_rows();
        rgs.push(RowGroupInfo {
            index: i,
            first_row: offset,
            num_rows: n,
        });
        offset += n;
    }
    let seg = SegmentFileInfo {
        writer_generation: 0,
        max_doc: NUM_ROWS as i64,
        object_path: object_store::path::Path::from(path.to_string_lossy().as_ref()),
        parquet_size: size,
        row_groups: rgs,
        arrow_schema: schema.clone(),
        metadata: parquet_meta,
        global_base: 0,
        sort_min: None,
        sort_max: None,
    };
    (seg, schema)
}

// ── Expressions ─────────────────────────────────────────────────────

fn col(schema: &Schema, name: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(PhysColumn::new(name, schema.index_of(name).unwrap()))
}

fn lit_str(v: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Utf8(Some(v.to_string()))))
}

fn lit_i32(v: i32) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int32(Some(v))))
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

/// `variant_get(v, path, ty)` as a physical expression bound to `schema`.
fn variant_get(schema: &Schema, path: &str, ty: &str) -> Arc<dyn PhysicalExpr> {
    let udf = Arc::new(ScalarUDF::from(VariantGetUdf::new()));
    Arc::new(
        ScalarFunctionExpr::try_new(
            udf,
            vec![col(schema, "v"), lit_str(path), lit_str(ty)],
            schema,
            Arc::new(ConfigOptions::default()),
        )
        .unwrap(),
    )
}

fn pred(expr: Arc<dyn PhysicalExpr>) -> BoolNode {
    BoolNode::Predicate(expr)
}

// ── Harness (mirrors page_pruning.rs, single all-docs collector) ────

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

fn collect_pred_exprs(node: &BoolNode, out: &mut Vec<Arc<dyn PhysicalExpr>>) {
    match node {
        BoolNode::Predicate(e) => out.push(Arc::clone(e)),
        BoolNode::And(cs) | BoolNode::Or(cs) => cs.iter().for_each(|c| collect_pred_exprs(c, out)),
        BoolNode::Not(c) => collect_pred_exprs(c, out),
        _ => {}
    }
}

fn build_pp_map(
    tree: &BoolNode,
    schema: &SchemaRef,
) -> Arc<HashMap<usize, Arc<datafusion::physical_optimizer::pruning::PruningPredicate>>> {
    let mut exprs = Vec::new();
    collect_pred_exprs(tree, &mut exprs);
    Arc::new(
        exprs
            .iter()
            .filter_map(|expr| {
                build_pruning_predicate(expr, schema.clone())
                    .map(|pp| (Arc::as_ptr(expr) as *const () as usize, pp))
            })
            .collect(),
    )
}

fn aggregate_metrics(plan: &Arc<dyn ExecutionPlan>) -> MetricsSet {
    let mut set = MetricsSet::new();
    fn walk(plan: &Arc<dyn ExecutionPlan>, out: &mut MetricsSet) {
        if plan.name() == "QueryShardExec" {
            if let Some(m) = plan.metrics() {
                for metric in m.iter() {
                    out.push(Arc::clone(metric));
                }
            }
        }
        for child in plan.children() {
            walk(child, out);
        }
    }
    walk(plan, &mut set);
    set
}

fn get_counter(set: &MetricsSet, name: &str) -> usize {
    use datafusion::physical_plan::metrics::MetricType;
    set.sum(|m| m.value().name() == name && m.metric_type() == MetricType::Dev)
        .map(|v| v.as_usize())
        .unwrap_or(0)
}

/// Runs `AND(all-docs collector, tree)` through the indexed table on the given fixture and
/// returns (matching ids sorted, plan, RG-level StatsPruneTree keep-vector).
async fn run(shredded: bool, tree: BoolNode) -> (Vec<i32>, Arc<dyn ExecutionPlan>, Vec<bool>) {
    let (tmp, _) = write_fixture(shredded);
    let (seg, schema) = load_segment(&tmp);
    let tree = BoolNode::And(vec![BoolNode::Collector { annotation_id: 0 }, tree]).push_not_down();
    let pp_map = build_pp_map(&tree, &schema);

    // RG-level decision exactly as table_provider.rs builds it per segment.
    let rg_indices: Vec<usize> = seg.row_groups.iter().map(|rg| rg.index).collect();
    let spt = StatsPruneTree::build_from_bool_node(
        &tree,
        &pp_map,
        &seg.metadata,
        &schema,
        &rg_indices,
        &seg.arrow_schema,
    );
    let rg_keep = spt.rg_can_match.clone();

    let per_leaf: Vec<(i32, Arc<dyn RowGroupDocsCollector>)> = vec![(0, Arc::new(AllDocs))];
    let tree = Arc::new(tree);
    let factory: EvaluatorFactory = {
        let tree = Arc::clone(&tree);
        let schema = schema.clone();
        let pp_map = Arc::clone(&pp_map);
        Arc::new(move |segment, _chunk, sm, _spt| {
            let resolved = tree.resolve(&per_leaf)?;
            let pruner = Arc::new(PagePruner::new(
                &schema,
                Arc::clone(&segment.metadata),
                schema.clone(),
            ));
            let eval: Arc<dyn RowGroupBitsetSource> = Arc::new(TreeBitsetSource {
                tree: Arc::new(resolved),
                evaluator: Arc::new(BitmapTreeEvaluator),
                leaves: Arc::new(CollectorLeafBitmaps::new(sm.ffm_collector_calls.clone())),
                page_pruner: pruner,
                cost_predicate: 1,
                cost_collector: 10,
                max_collector_parallelism: 1,
                pruning_predicates: Arc::clone(&pp_map),
                page_prune_metrics: Some(
                    crate::indexed_table::page_pruner::PagePruneMetrics::from_stream_metrics(sm),
                ),
                collector_strategy: CollectorCallStrategy::TightenOuterBounds,
                stats_prune_tree: None,
                rg_index_to_pos: HashMap::new(),
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
    let provider = Arc::new(IndexedTableProvider::new(IndexedTableConfig {
        schema: schema.clone(),
        segments: vec![seg],
        store,
        store_url,
        evaluator_factory: factory,
        pushdown_predicate: None,
        query_config: Arc::new(qc),
        // The refinement stage evaluates the predicate on the scanned batch; `v` is not in the
        // SELECT list so it has to be pulled in as a predicate column.
        predicate_columns: vec![schema.index_of("v").unwrap()],
        emit_row_ids: false,
        prune_tree_config: None,
        sort_fields: vec![],
        sort_orders: vec![],
        cancellation_token: None,
    }));
    let ctx = SessionContext::new();
    ctx.register_table("t", provider).unwrap();
    let df = ctx.sql("SELECT id FROM t").await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let mut stream =
        datafusion::physical_plan::execute_stream(Arc::clone(&plan), ctx.task_ctx()).unwrap();
    let mut ids = Vec::new();
    while let Some(batch) = stream.next().await {
        let b = batch.unwrap();
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        ids.extend(c.iter().flatten());
    }
    ids.sort();
    (ids, plan, rg_keep)
}

fn expected_ids(f: impl Fn(usize) -> bool) -> Vec<i32> {
    (0..NUM_ROWS).filter(|r| f(*r)).map(|r| r as i32).collect()
}

// ═════════════════════════════════════════════════════════════════════
// Step 2: placement
// ═════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn placement_pushes_variant_get_to_projection_above_scan() {
    // Placement is a logical-plan property; a MemTable carrying the footer-equivalent schema
    // (with the extension tag) is enough to observe ExtractLeafExpressions at work.
    let variant = variant_column(false);
    let v_field = variant.field("v");
    let v: ArrayRef = Arc::new(variant.into_inner());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        v_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from((0..NUM_ROWS as i32).collect::<Vec<_>>())),
            v,
        ],
    )
    .unwrap();
    let ctx = SessionContext::new();
    crate::udf::variant_get::register_all(&ctx);
    ctx.register_batch("t", batch).unwrap();
    let plan = ctx
        .sql("SELECT variant_get(v, 'x', 'Int64') AS x FROM t WHERE id < 10 ORDER BY id")
        .await
        .unwrap()
        .into_optimized_plan()
        .unwrap();

    fn find_extraction_above_scan(plan: &LogicalPlan) -> bool {
        if let LogicalPlan::Projection(p) = plan {
            let has_udf = p
                .expr
                .iter()
                .any(|e| format!("{e}").contains("variant_get"));
            if has_udf && matches!(p.input.as_ref(), LogicalPlan::TableScan(_)) {
                return true;
            }
        }
        plan.inputs().iter().any(|i| find_extraction_above_scan(i))
    }
    assert!(
        find_extraction_above_scan(&plan),
        "expected variant_get in a Projection directly above TableScan, got:\n{}",
        plan.display_indent()
    );

    let batches = ctx
        .sql("SELECT variant_get(v, 'x', 'Int64') AS x FROM t WHERE id < 3 ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let xs: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(xs, vec![0, 1, 2]);
}

/// V-27: DataFusion's ListingTable schema inference rebuilds the struct children (BinaryView,
/// non-null) and drops the parent's extension metadata, so `variant_get` refuses the column at
/// plan time. Our IndexedTable takes the footer schema verbatim and is unaffected, but any
/// schema-transforming path (view-type coercion, union schema adapters for D-18) must carry
/// the tag through explicitly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn listing_table_inference_drops_extension_tag() {
    let (tmp, _) = write_fixture(false);
    let ctx = SessionContext::new();
    crate::udf::variant_get::register_all(&ctx);
    let opts = ParquetReadOptions {
        file_extension: "",
        ..ParquetReadOptions::default()
    };
    ctx.register_parquet("t", tmp.path().to_str().unwrap(), opts)
        .await
        .unwrap();
    let inferred = ctx.table("t").await.unwrap().schema().clone();
    let v = inferred.field_with_unqualified_name("v").unwrap();
    assert!(
        !v.metadata().contains_key("ARROW:extension:name"),
        "if this starts passing, ListingTable now preserves the tag and V-27 can be closed: {v:?}"
    );
    let err = ctx
        .sql("SELECT variant_get(v, 'x', 'Int64') FROM t")
        .await
        .err()
        .expect("variant_get must refuse an untagged struct");
    assert!(
        format!("{err}").contains("must be a Variant column"),
        "{err}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Step 2b: unshredded predicate never prunes, results exact (V-24)
// ═════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unshredded_predicate_alone_prunes_nothing() {
    let (_, schema) = write_fixture(false);
    let expr = binop(
        variant_get(&schema, "x", "Int64"),
        Operator::Lt,
        lit_i64(1024),
    );
    let (ids, plan, rg_keep) = run(false, pred(expr)).await;
    assert_eq!(ids, expected_ids(|r| x_value(r) < 1024));
    assert_eq!(rg_keep, vec![true]);
    let m = aggregate_metrics(&plan);
    assert_eq!(get_counter(&m, "pages_total"), NUM_PAGES);
    assert_eq!(
        get_counter(&m, "pages_pruned"),
        0,
        "opaque UDF predicate must not prune pages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unshredded_predicate_under_not_prunes_nothing() {
    let (_, schema) = write_fixture(false);
    let expr = binop(
        variant_get(&schema, "x", "Int64"),
        Operator::Lt,
        lit_i64(1024),
    );
    let (ids, plan, rg_keep) = run(false, BoolNode::Not(Box::new(pred(expr)))).await;
    assert_eq!(ids, expected_ids(|r| x_value(r) >= 1024));
    assert_eq!(rg_keep, vec![true]);
    assert_eq!(get_counter(&aggregate_metrics(&plan), "pages_pruned"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unshredded_predicate_in_or_with_typed_prunes_nothing() {
    let (_, schema) = write_fixture(false);
    let typed = binop(col(&schema, "id"), Operator::Lt, lit_i32(1024));
    let opaque = binop(
        variant_get(&schema, "x", "Int64"),
        Operator::GtEq,
        lit_i64(30_000),
    );
    let (ids, plan, rg_keep) = run(false, BoolNode::Or(vec![pred(typed), pred(opaque)])).await;
    assert_eq!(ids, expected_ids(|r| r < 1024 || x_value(r) >= 30_000));
    assert_eq!(rg_keep, vec![true]);
    assert_eq!(get_counter(&aggregate_metrics(&plan), "pages_pruned"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unshredded_predicate_in_and_with_typed_prunes_by_typed_only() {
    let (_, schema) = write_fixture(false);
    let typed = binop(col(&schema, "id"), Operator::Lt, lit_i32(1024));
    let opaque = binop(
        variant_get(&schema, "x", "Int64"),
        Operator::Lt,
        lit_i64(500),
    );
    let (ids, plan, rg_keep) = run(false, BoolNode::And(vec![pred(typed), pred(opaque)])).await;
    assert_eq!(ids, expected_ids(|r| r < 1024 && x_value(r) < 500));
    assert_eq!(rg_keep, vec![true]);
    let m = aggregate_metrics(&plan);
    assert_eq!(
        get_counter(&m, "pages_pruned"),
        3,
        "the typed conjunct alone must still prune"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Steps 3-5: shredded adversarial file
// ═════════════════════════════════════════════════════════════════════

#[test]
fn shredded_fixture_has_expected_leaves_and_flat_view() {
    let (tmp, schema) = write_fixture(true);
    let (seg, _) = load_segment(&tmp);
    let descr = seg.metadata.file_metadata().schema_descr();
    let paths: Vec<String> = descr.columns().iter().map(|c| c.path().string()).collect();
    assert_eq!(
        paths,
        vec![
            "id",
            "v.metadata",
            "v.value",
            "v.typed_value.x.value",
            "v.typed_value.x.typed_value",
        ]
    );
    assert_eq!(seg.metadata.num_row_groups(), 4);

    // Stock converter cannot see nested leaves (this is what V-17 asked).
    use datafusion::parquet::arrow::arrow_reader::statistics::StatisticsConverter;
    assert!(StatisticsConverter::try_new("v.typed_value.x.typed_value", &schema, descr).is_err());

    // The flat view can, and reads the right stats: RG 2 typed min/max exclude the fallback
    // row, and its value leaf has exactly one non-null.
    let view = FlatLeafView::build(&schema, descr).expect("nested leaves present");
    let typed = view.converter("v.typed_value.x.typed_value").unwrap();
    let rgs: Vec<_> = seg.metadata.row_groups().iter().collect();
    let mins = typed.row_group_mins(rgs.iter().copied()).unwrap();
    let maxes = typed.row_group_maxes(rgs.iter().copied()).unwrap();
    let mins = mins.as_any().downcast_ref::<Int64Array>().unwrap();
    let maxes = maxes.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(mins.values(), &[0, 10_000, 20_000, 30_000]);
    assert_eq!(maxes.values(), &[1023, 11_023, 21_023, 31_023]);
    let value = view.converter("v.typed_value.x.value").unwrap();
    let nulls = value.row_group_null_counts(rgs.iter().copied()).unwrap();
    assert_eq!(nulls.values(), &[1024, 1024, 1023, 1024]);
}

/// The hazard and the guard, at RG level (StatsPruneTree) and page level (PagePruner).
#[test]
fn shredded_naive_rewrite_prunes_fallback_rg_but_or_guard_keeps_it() {
    let (tmp, _) = write_fixture(true);
    let (seg, seg_schema) = load_segment(&tmp);
    let descr = seg.metadata.file_metadata().schema_descr();
    let view = FlatLeafView::build(&seg_schema, descr).unwrap();
    let flat: SchemaRef = Arc::new(view.schema().clone());
    let rg_indices = vec![0usize, 1, 2, 3];

    let typed_leaf = col(&flat, "v.typed_value.x.typed_value");
    let value_leaf = col(&flat, "v.typed_value.x.value");

    // Naive: rewrite variant_get(v,'x','Int64') < 1024 to the typed leaf only.
    let naive = binop(Arc::clone(&typed_leaf), Operator::Lt, lit_i64(1024));
    let tree = pred(Arc::clone(&naive));
    let pp = build_pp_map(&tree, &flat);
    let spt = StatsPruneTree::build_from_bool_node(
        &tree,
        &pp,
        &seg.metadata,
        &flat,
        &rg_indices,
        &seg_schema,
    );
    assert_eq!(
        spt.rg_can_match,
        vec![true, false, false, false],
        "naive rewrite prunes RG 2 although it holds a fallback row: the hazard D-08 guards against"
    );

    // Guarded: typed < c OR value IS NOT NULL.
    let guarded = Arc::new(BinaryExpr::new(
        naive,
        Operator::Or,
        is_not_null(Arc::clone(&value_leaf)).unwrap(),
    )) as Arc<dyn PhysicalExpr>;
    let tree = pred(Arc::clone(&guarded));
    let pp = build_pp_map(&tree, &flat);
    let spt = StatsPruneTree::build_from_bool_node(
        &tree,
        &pp,
        &seg.metadata,
        &flat,
        &rg_indices,
        &seg_schema,
    );
    assert_eq!(
        spt.rg_can_match,
        vec![true, false, true, false],
        "guard keeps RG 0 (typed match) and RG 2 (fallback row), prunes RG 1 and 3"
    );

    // Page level inside RG 2: only page 1 (rows 256..512) holds the fallback row.
    let pruner = PagePruner::new(&flat, Arc::clone(&seg.metadata), seg_schema.clone());
    let pp_guarded = build_pruning_predicate(&guarded, flat.clone()).unwrap();
    let sel = pruner
        .prune_rg(&pp_guarded, 2, None)
        .expect("page index present");
    let selected: Vec<(usize, bool)> = sel.iter().map(|s| (s.row_count, !s.skip)).collect();
    let kept_rows: usize = selected
        .iter()
        .filter(|(_, keep)| *keep)
        .map(|(n, _)| n)
        .sum();
    assert_eq!(
        kept_rows, SHREDDED_PAGE_ROWS,
        "exactly one page of RG 2 survives: {selected:?}"
    );
    let mut offset = 0usize;
    let mut kept_range = None;
    for (n, keep) in &selected {
        if *keep {
            kept_range = Some(offset..offset + n);
        }
        offset += n;
    }
    assert!(kept_range
        .unwrap()
        .contains(&(FALLBACK_ROW - 2 * ROWS_PER_PAGE)));
}

/// End to end through the indexed table on the shredded file with the real UDF predicate:
/// exact results (brute force via the same kernel), nothing pruned by the opaque predicate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shredded_variant_get_predicate_end_to_end_is_exact() {
    let (_, schema) = write_fixture(true);
    // What does the kernel say about the fallback row under Int64 with safe cast?
    let variant = variant_column(true);
    let v: ArrayRef = Arc::new(variant.into_inner());
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from((0..NUM_ROWS as i32).collect::<Vec<_>>())),
            v,
        ],
    )
    .unwrap();
    let extracted = variant_get(&schema, "x", "Int64")
        .evaluate(&batch)
        .unwrap()
        .into_array(NUM_ROWS)
        .unwrap();
    let extracted = extracted.as_any().downcast_ref::<Int64Array>().unwrap();
    let fallback_as_i64: Option<i64> = if extracted.is_null(FALLBACK_ROW) {
        None
    } else {
        Some(extracted.value(FALLBACK_ROW))
    };
    let brute: Vec<i32> = (0..NUM_ROWS)
        .filter(|r| !extracted.is_null(*r) && extracted.value(*r) < 1024)
        .map(|r| r as i32)
        .collect();

    let expr = binop(
        variant_get(&schema, "x", "Int64"),
        Operator::Lt,
        lit_i64(1024),
    );
    let (ids, plan, rg_keep) = run(true, pred(expr)).await;
    assert_eq!(ids, brute, "indexed-table results must equal brute force (fallback row as Int64 = {fallback_as_i64:?})");
    assert_eq!(rg_keep, vec![true, true, true, true]);
    assert_eq!(get_counter(&aggregate_metrics(&plan), "pages_pruned"), 0);
}
