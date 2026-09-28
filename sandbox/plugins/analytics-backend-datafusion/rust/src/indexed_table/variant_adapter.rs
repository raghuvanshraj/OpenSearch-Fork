/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! D-18(a): per-segment Variant layout, one output type.
//!
//! Shredding is a per-file physical layout. `build_segments` gives the table one canonical
//! `Struct<metadata, value>` type for every Variant column, and each segment is read against its
//! own footer schema. Bridging the two is normally DataFusion's `DefaultPhysicalExprAdapter`,
//! which turns a file's `Struct<metadata, value, typed_value>` into the table's two-field struct
//! with a name-based struct cast — that would *drop* `typed_value`, i.e. silently lose every
//! shredded value. [`VariantExprAdapterFactory`] runs after the default adapter and swaps that
//! cast for the spec-defined reconstruction (`unshred_variant`), so a segment shredded on any
//! path set, or not at all, produces the same canonical column.
//!
//! Only `Column`s whose logical *and* physical fields carry the `arrow.parquet.variant` extension
//! tag are touched; every other rewrite is left to the default adapter.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::Result;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::expressions::{CastExpr, Column};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr_adapter::{
    DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter, PhysicalExprAdapterFactory,
};
use parquet::variant::{unshred_variant, VariantArray, VariantType};

fn is_variant(field: &Field) -> bool {
    field.try_extension_type::<VariantType>().is_ok()
}

/// Factory installed on every `FileScanConfig` the indexed table builds.
#[derive(Debug, Clone, Default)]
pub struct VariantExprAdapterFactory;

impl PhysicalExprAdapterFactory for VariantExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        let inner = DefaultPhysicalExprAdapterFactory
            .create(logical_file_schema, Arc::clone(&physical_file_schema))?;
        Ok(Arc::new(VariantExprAdapter {
            inner,
            physical_file_schema,
        }))
    }
}

#[derive(Debug)]
struct VariantExprAdapter {
    inner: Arc<dyn PhysicalExprAdapter>,
    physical_file_schema: SchemaRef,
}

impl PhysicalExprAdapter for VariantExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        let expr = self.inner.rewrite(expr)?;
        expr.transform(|e| {
            let Some(cast) = e.downcast_ref::<CastExpr>() else {
                return Ok(Transformed::no(e));
            };
            if !is_variant(cast.target_field()) {
                return Ok(Transformed::no(e));
            }
            let Some(col) = cast.expr().downcast_ref::<Column>() else {
                return Ok(Transformed::no(e));
            };
            let physical_is_variant = self
                .physical_file_schema
                .field_with_name(col.name())
                .map(is_variant)
                .unwrap_or(false);
            if !physical_is_variant {
                return Ok(Transformed::no(e));
            }
            Ok(Transformed::yes(Arc::new(UnshredVariantExpr::new(
                Arc::clone(cast.expr()),
                Arc::clone(cast.target_field()),
            ))))
        })
        .data()
    }
}

/// Reconstructs a Variant column read in any shredded layout into the table's canonical
/// `Struct<metadata, value>` type. `unshred_variant` is the spec's reconstruction: `typed_value`
/// is folded back into `value`, partially shredded objects are merged with their residual, and
/// rows whose `typed_value` is NULL are taken from `value` as-is.
#[derive(Debug, Eq)]
pub struct UnshredVariantExpr {
    input: Arc<dyn PhysicalExpr>,
    target_field: FieldRef,
}

impl PartialEq for UnshredVariantExpr {
    fn eq(&self, other: &Self) -> bool {
        self.input.eq(&other.input) && self.target_field == other.target_field
    }
}

impl Hash for UnshredVariantExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.target_field.hash(state);
    }
}

impl UnshredVariantExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, target_field: FieldRef) -> Self {
        Self {
            input,
            target_field,
        }
    }
}

impl fmt::Display for UnshredVariantExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unshred_variant({})", self.input)
    }
}

impl PhysicalExpr for UnshredVariantExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.target_field.data_type().clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(self.target_field.is_nullable())
    }

    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::clone(&self.target_field))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let arr = self.input.evaluate(batch)?.into_array(batch.num_rows())?;
        let va = VariantArray::try_new(arr.as_ref())?;
        let inner: ArrayRef = if va.typed_value_field().is_some() {
            Arc::new(unshred_variant(&va)?.into_inner())
        } else {
            arr
        };
        let out = if inner.data_type() == self.target_field.data_type() {
            inner
        } else {
            datafusion::arrow::compute::cast(&inner, self.target_field.data_type())?
        };
        Ok(ColumnarValue::Array(out))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(UnshredVariantExpr::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.target_field),
        )))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unshred_variant(")?;
        self.input.fmt_sql(f)?;
        write!(f, ")")
    }
}
