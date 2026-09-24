/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.fields.core.data.variant;

import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.opensearch.index.engine.dataformat.FieldTypeCapabilities;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.parquet.fields.ParquetField;
import org.opensearch.parquet.vsr.ManagedVSR;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Parquet field storing a Parquet Variant value (semi-structured data) in its unshredded form.
 * <p>
 * The Arrow representation is the canonical Variant extension type: a
 * {@code Struct<metadata: Binary, value: Binary>} whose {@link Field} carries
 * {@code ARROW:extension:name = arrow.parquet.variant}. The two children hold the Variant
 * binary encoding produced by {@link VariantEncoder}. The extension tag rides the C Data
 * Interface as field metadata; arrow-rs (with the {@code variant_experimental} feature)
 * recognises it and annotates the Parquet group with the {@code VARIANT} logical type.
 * <p>
 * The exact shape matters: arrow-rs's writer checks only that the tagged type is a struct
 * ({@code VariantType::supports_data_type}), while readers ({@code VariantArray::try_new})
 * check the children. A mis-shaped struct would therefore be written as a {@code VARIANT}
 * group and rejected on read. {@link #validateShape(Field)} enforces the canonical shape on
 * the Java side so that mistake cannot reach a segment.
 */
public class VariantParquetField extends ParquetField {

    /** Mapper content type this field serves. */
    public static final String CONTENT_TYPE = "variant";
    /** Arrow field-metadata key naming the extension type. */
    public static final String EXTENSION_NAME_KEY = "ARROW:extension:name";
    /** Arrow field-metadata key carrying extension-specific metadata (empty for Variant). */
    public static final String EXTENSION_METADATA_KEY = "ARROW:extension:metadata";
    /** Canonical Arrow extension name for Variant; must match arrow-rs {@code VariantType::NAME}. */
    public static final String VARIANT_EXTENSION_NAME = "arrow.parquet.variant";
    /** Name of the struct child holding the Variant metadata (dictionary) bytes. */
    public static final String METADATA_CHILD = "metadata";
    /** Name of the struct child holding the Variant value bytes. */
    public static final String VALUE_CHILD = "value";

    /**
     * Encoded Variant for one document: the metadata (dictionary) blob and the value blob.
     *
     * @param metadata Variant metadata bytes
     * @param value Variant value bytes
     */
    public record VariantBytes(byte[] metadata, byte[] value) {
    }

    /** Creates a new VariantParquetField. */
    public VariantParquetField() {}

    @Override
    protected void addToGroup(MappedFieldType fieldType, ManagedVSR managedVSR, Object parseValue) {
        StructVector struct = (StructVector) managedVSR.getVector(fieldType.name());
        int row = managedVSR.getRowCount();
        writeRow(struct, row, parseValue);
    }

    /**
     * Writes one Variant (or null) at {@code row} of {@code struct}.
     *
     * @param struct the Variant struct vector
     * @param row the row index
     * @param parseValue a {@link VariantBytes}, or null for a missing field
     */
    public static void writeRow(StructVector struct, int row, Object parseValue) {
        VarBinaryVector metadata = (VarBinaryVector) struct.getChild(METADATA_CHILD);
        VarBinaryVector value = (VarBinaryVector) struct.getChild(VALUE_CHILD);
        if (parseValue == null) {
            struct.setNull(row);
            metadata.setNull(row);
            value.setNull(row);
            return;
        }
        VariantBytes bytes = (VariantBytes) parseValue;
        struct.setIndexDefined(row);
        metadata.setSafe(row, bytes.metadata());
        value.setSafe(row, bytes.value());
    }

    @Override
    public ArrowType getArrowType() {
        return ArrowType.Struct.INSTANCE;
    }

    @Override
    public FieldType getFieldType() {
        return new FieldType(true, getArrowType(), null, Map.of(EXTENSION_NAME_KEY, VARIANT_EXTENSION_NAME, EXTENSION_METADATA_KEY, ""));
    }

    @Override
    protected List<Field> getChildFields() {
        // metadata is required per the Variant spec; value is nullable so the shredded form
        // (typed_value present, value absent for a row) uses the same child definition.
        return List.of(
            new Field(METADATA_CHILD, FieldType.notNullable(new ArrowType.Binary()), null),
            new Field(VALUE_CHILD, FieldType.nullable(new ArrowType.Binary()), null)
        );
    }

    @Override
    public Set<FieldTypeCapabilities.Capability> supportedCapabilities() {
        return Set.of(FieldTypeCapabilities.Capability.COLUMNAR_STORAGE, FieldTypeCapabilities.Capability.STORED_FIELDS);
    }

    /**
     * Asserts that {@code field} has exactly the canonical unshredded Variant shape this class
     * emits. Throws {@link IllegalArgumentException} describing the first deviation.
     * <p>
     * This is the Java half of the write-side shape guard (design decision D-17): arrow-rs
     * will annotate any tagged struct as {@code VARIANT}, so shape mistakes must be caught
     * before the schema crosses the FFI boundary.
     *
     * @param field the Arrow field to check
     */
    public static void validateShape(Field field) {
        if (VARIANT_EXTENSION_NAME.equals(field.getMetadata().get(EXTENSION_NAME_KEY)) == false) {
            throw new IllegalArgumentException(
                "Variant field [" + field.getName() + "] missing extension name [" + VARIANT_EXTENSION_NAME + "]"
            );
        }
        if (field.getType().getTypeID() != ArrowType.ArrowTypeID.Struct) {
            throw new IllegalArgumentException("Variant field [" + field.getName() + "] must be a Struct, got " + field.getType());
        }
        List<Field> children = field.getChildren();
        if (children.size() != 2) {
            throw new IllegalArgumentException(
                "Variant field [" + field.getName() + "] must have exactly [metadata, value] children, got " + children
            );
        }
        requireBinaryChild(field, children.get(0), METADATA_CHILD, false);
        requireBinaryChild(field, children.get(1), VALUE_CHILD, true);
    }

    private static void requireBinaryChild(Field parent, Field child, String expectedName, boolean nullable) {
        if (expectedName.equals(child.getName()) == false) {
            throw new IllegalArgumentException(
                "Variant field [" + parent.getName() + "] child [" + child.getName() + "] must be named [" + expectedName + "]"
            );
        }
        if (child.getType().getTypeID() != ArrowType.ArrowTypeID.Binary) {
            throw new IllegalArgumentException(
                "Variant field [" + parent.getName() + "] child [" + expectedName + "] must be Binary, got " + child.getType()
            );
        }
        if (child.isNullable() != nullable) {
            throw new IllegalArgumentException(
                "Variant field [" + parent.getName() + "] child [" + expectedName + "] nullable must be " + nullable
            );
        }
    }
}
