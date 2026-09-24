/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

//! `variant_get(v, path [, type])`: extract a path from a Parquet Variant column.
//!
//! * `v` must be a column carrying the `arrow.parquet.variant` extension type (validated at
//!   plan time in [`ScalarUDFImpl::return_field_from_args`]).
//! * `path` is a string literal in dot/bracket notation (`user.id`, `a['b.c'].d[2]`); a leading
//!   `$.` is accepted and ignored.
//! * `type`, when present, is a string literal naming the Arrow output type (`Int64`, `Int32`,
//!   `Float64`, `Utf8`, `Boolean`). Rows whose value cannot be cast become NULL (safe cast).
//!   Without `type` the result is the extracted sub-variant serialised to JSON text: Variant
//!   values never leave the Rust side as an extension type in v1.
//!
//! Placement is [`ExpressionPlacement::MoveTowardsLeafNodes`] so DataFusion's
//! `ExtractLeafExpressions` pushes the call into a projection directly above the scan.
//!
//! POC-2 status: unshredded and shredded reads both go through `parquet_variant_compute::variant_get`,
//! which follows `typed_value` when present. The stats-soundness guard and the nested-leaf
//! projection rule live outside this file.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::extension::ExtensionType;
use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::compute::CastOptions;
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{exec_err, plan_err, Result, ScalarValue};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{
    ColumnarValue, ExpressionPlacement, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use parquet::variant::{
    unshred_variant, variant_get, variant_to_json, GetOptions, VariantArray, VariantPath,
    VariantType,
};

pub const NAME: &str = "variant_get";

/// Serialise a Variant array to JSON text.
///
/// `parquet_variant_compute::variant_to_json` at 58.3.0 only accepts the unshredded layout with
/// `Binary` (not `BinaryView`) children, while the kernels return `BinaryView` and shredded
/// results carry `typed_value`. Normalise first: unshred if needed, then cast children to
/// `Binary`. Both steps are no-ops on the common read-from-parquet layout.
pub fn variant_to_json_text(arr: &ArrayRef) -> Result<datafusion::arrow::array::StringArray> {
    let mut va = VariantArray::try_new(arr.as_ref())?;
    if va.typed_value_field().is_some() {
        va = unshred_variant(&va)?;
    }
    let inner: ArrayRef = Arc::new(va.into_inner());
    let canonical = DataType::Struct(
        vec![
            Arc::new(Field::new("metadata", DataType::Binary, false)),
            Arc::new(Field::new("value", DataType::Binary, true)),
        ]
        .into(),
    );
    let inner = if inner.data_type() == &canonical {
        inner
    } else {
        datafusion::arrow::compute::cast(&inner, &canonical)?
    };
    Ok(variant_to_json(&inner)?)
}

pub fn register_all(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::from(VariantGetUdf::new()));
}

#[derive(Debug)]
pub struct VariantGetUdf {
    signature: Signature,
}

impl VariantGetUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(2), TypeSignature::Any(3)],
                Volatility::Immutable,
            ),
        }
    }
}

impl Default for VariantGetUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for VariantGetUdf {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for VariantGetUdf {}
impl Hash for VariantGetUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        NAME.hash(state);
    }
}

/// Parse the `type` literal. Kept deliberately small for the POC.
fn parse_type_name(name: &str) -> Result<DataType> {
    Ok(match name {
        "Int64" | "BIGINT" | "bigint" => DataType::Int64,
        "Int32" | "INT" | "int" => DataType::Int32,
        "Float64" | "DOUBLE" | "double" => DataType::Float64,
        "Utf8" | "VARCHAR" | "varchar" | "string" => DataType::Utf8,
        "Boolean" | "BOOLEAN" | "boolean" => DataType::Boolean,
        other => return plan_err!("{NAME}: unsupported target type '{other}'"),
    })
}

fn literal_str<'a>(sv: Option<&'a ScalarValue>, what: &str) -> Result<&'a str> {
    match sv {
        Some(ScalarValue::Utf8(Some(s))) | Some(ScalarValue::LargeUtf8(Some(s))) => Ok(s),
        Some(other) => plan_err!("{NAME}: {what} must be a string literal, got {other}"),
        None => plan_err!("{NAME}: {what} must be a literal"),
    }
}

fn strip_root(path: &str) -> &str {
    path.strip_prefix("$.")
        .or_else(|| path.strip_prefix('$'))
        .unwrap_or(path)
}

impl ScalarUDFImpl for VariantGetUdf {
    fn name(&self) -> &str {
        NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Never called: return_field_from_args is implemented.
        plan_err!("{NAME}: return_type called directly")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let input = &args.arg_fields[0];
        if input.try_extension_type::<VariantType>().is_err() {
            return plan_err!(
                "{NAME}: first argument must be a Variant column (extension type '{}'), got field '{}' of type {}",
                VariantType::NAME,
                input.name(),
                input.data_type()
            );
        }
        let path = literal_str(args.scalar_arguments[1], "path")?;
        VariantPath::try_from(strip_root(path)).map_err(|e| {
            datafusion::common::DataFusionError::Plan(format!("{NAME}: bad path '{path}': {e}"))
        })?;
        let out_type = if args.arg_fields.len() == 3 {
            parse_type_name(literal_str(args.scalar_arguments[2], "type")?)?
        } else {
            DataType::Utf8
        };
        Ok(Arc::new(Field::new(NAME, out_type, true)))
    }

    fn placement(&self, _args: &[ExpressionPlacement]) -> ExpressionPlacement {
        ExpressionPlacement::MoveTowardsLeafNodes
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;
        let input: ArrayRef = match &args.args[0] {
            ColumnarValue::Array(a) => Arc::clone(a),
            ColumnarValue::Scalar(s) => s.to_array_of_size(num_rows)?,
        };
        let path = match &args.args[1] {
            ColumnarValue::Scalar(sv) => literal_str(Some(sv), "path")?.to_string(),
            ColumnarValue::Array(_) => return exec_err!("{NAME}: path must be a literal"),
        };
        let as_type = if args.args.len() == 3 {
            match &args.args[2] {
                ColumnarValue::Scalar(sv) => Some(parse_type_name(literal_str(Some(sv), "type")?)?),
                ColumnarValue::Array(_) => return exec_err!("{NAME}: type must be a literal"),
            }
        } else {
            None
        };

        let variant_path = VariantPath::try_from(strip_root(&path)).map_err(|e| {
            datafusion::common::DataFusionError::Execution(format!("{NAME}: bad path: {e}"))
        })?;
        let options = GetOptions::new_with_path(variant_path)
            .with_as_type(
                as_type
                    .clone()
                    .map(|dt| Arc::new(Field::new("v", dt, true))),
            )
            .with_cast_options(CastOptions {
                safe: true,
                ..Default::default()
            });

        let extracted = variant_get(&input, options)?;
        let out: ArrayRef = if as_type.is_some() {
            extracted
        } else {
            // v1 result contract: Variant-typed results become JSON text at the Rust boundary.
            Arc::new(variant_to_json_text(&extracted)?)
        };
        debug_assert_eq!(out.len(), num_rows);
        Ok(ColumnarValue::Array(out))
    }
}
