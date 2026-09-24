/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! Statistics access for nested parquet leaves (POC-2, design decision D-09).
//!
//! `StatisticsConverter::try_new(name, arrow_schema, descr)` resolves `name` against the
//! top-level arrow fields and refuses nested types (`parquet_column` returns `None` for any
//! field whose type `is_nested()`), so a predicate on a shredded Variant leaf such as
//! `v.typed_value.x.typed_value` gets no statistics and can never prune.
//!
//! [`FlatLeafView`] sidesteps that without touching parquet-rs: it builds a synthetic arrow
//! schema whose fields are the file's parquet *leaves*, named by their full dotted path, and
//! a synthetic `SchemaDescriptor` with the same leaves re-rooted as top-level primitives in
//! the same order. In that pair, leaf `i` is root `i`, so `parquet_column` maps the flat
//! field straight to parquet column index `i`, which is exactly the index the real
//! `RowGroupMetaData` / offset index use. Statistics decoding needs only the physical type
//! (cloned from the real leaf) and the arrow type (walked from the real arrow schema), both
//! of which are preserved.
//!
//! Limitations (acceptable for the POC): only struct nesting is walked (no `LIST`
//! `.list.element` paths); a leaf whose arrow type cannot be resolved is omitted, which
//! makes the predicate "unknown" for that leaf (conservative).

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use datafusion::parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

/// Synthetic flat view over a file's parquet leaves. See module docs.
pub struct FlatLeafView {
    schema: Schema,
    descr: SchemaDescriptor,
}

impl FlatLeafView {
    /// Builds the view. Returns `None` when the file has no nested leaves (nothing to add over
    /// the normal top-level path) or the synthetic descriptor cannot be built.
    pub fn build(seg_arrow_schema: &Schema, descr: &SchemaDescriptor) -> Option<Self> {
        let leaves = descr.columns();
        if leaves.iter().all(|c| c.path().parts().len() == 1) {
            return None;
        }
        let mut flat_fields = Vec::with_capacity(leaves.len());
        let mut flat_types = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let parts = leaf.path().parts();
            let name = leaf.path().string();
            // Arrow type: walk the real arrow schema through struct children. Unknown -> Null
            // placeholder, which makes StatisticsConverter fail for that leaf (conservative).
            let arrow_type = leaf_arrow_type(seg_arrow_schema, parts).unwrap_or(DataType::Null);
            flat_fields.push(Field::new(name, arrow_type, true));
            flat_types.push(Arc::new(leaf.self_type().clone()));
        }
        let root = ParquetType::group_type_builder("schema")
            .with_fields(flat_types)
            .build()
            .ok()?;
        Some(Self {
            schema: Schema::new(flat_fields),
            descr: SchemaDescriptor::new(Arc::new(root)),
        })
    }

    /// A converter for the leaf named by its full dotted path, or `None` if unknown.
    pub fn converter<'a>(&'a self, leaf_path: &str) -> Option<StatisticsConverter<'a>> {
        let (_, field) = self.schema.column_with_name(leaf_path)?;
        if field.data_type() == &DataType::Null {
            return None;
        }
        StatisticsConverter::try_new(leaf_path, &self.schema, &self.descr).ok()
    }

    /// The synthetic flat schema (one top-level field per parquet leaf, dotted names).
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// True when `name` looks like a nested leaf path this view could serve.
    pub fn is_candidate(name: &str) -> bool {
        name.contains('.')
    }
}

fn leaf_arrow_type(schema: &Schema, parts: &[String]) -> Option<DataType> {
    let (first, rest) = parts.split_first()?;
    let mut field: &Field = schema.field_with_name(first).ok()?;
    for part in rest {
        match field.data_type() {
            DataType::Struct(children) => {
                field = children.iter().find(|c| c.name() == part)?.as_ref();
            }
            _ => return None,
        }
    }
    if field.data_type().is_nested() {
        return None;
    }
    Some(field.data_type().clone())
}
