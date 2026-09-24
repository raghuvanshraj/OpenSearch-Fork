/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.fields.core.data.variant;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.parquet.variant.Variant;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.nativebridge.spi.ArrowExport;
import org.opensearch.parquet.bridge.NativeParquetWriter;
import org.opensearch.parquet.bridge.ParquetSortConfig;
import org.opensearch.parquet.bridge.RustBridge;
import org.opensearch.parquet.fields.core.data.variant.VariantParquetField.VariantBytes;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * POC-1: prove the unshredded Variant write path on the current pins (arrow-rs 58.3.0 +
 * {@code variant_experimental}, Arrow Java 18.1.0, parquet-variant 1.17.1) and the read-side
 * acceptance of what it writes.
 */
public class VariantParquetFieldTests extends OpenSearchTestCase {

    private static final String DOC_1 = "{\"user\":{\"id\":7,\"tags\":[\"x\",\"yz\"]},\"id\":42}";
    private static final String DOC_2 = "{\"user\":{\"id\":\"seven\",\"name\":\"Ada\"},\"id\":43,\"flag\":true,\"ratio\":0.5}";
    private static final String DOC_3 =
        "{\"id\":44,\"nested\":{\"deep\":{\"deeper\":[1,2,{\"k\":null}]}},\"big\":123456789012345678901234567890}";

    private BufferAllocator allocator;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        RustBridge.initLogger();
        allocator = new RootAllocator();
    }

    @Override
    public void tearDown() throws Exception {
        allocator.close();
        super.tearDown();
    }

    // ---- shape ----

    public void testArrowFieldIsCanonicalVariantShape() {
        Field field = new VariantParquetField().toArrowField("v", false);
        assertEquals(ArrowType.Struct.INSTANCE, field.getType());
        assertEquals(VariantParquetField.VARIANT_EXTENSION_NAME, field.getMetadata().get(VariantParquetField.EXTENSION_NAME_KEY));
        assertEquals("", field.getMetadata().get(VariantParquetField.EXTENSION_METADATA_KEY));
        assertEquals(2, field.getChildren().size());
        assertEquals("metadata", field.getChildren().get(0).getName());
        assertFalse(field.getChildren().get(0).isNullable());
        assertEquals("value", field.getChildren().get(1).getName());
        assertTrue(field.getChildren().get(1).isNullable());
        VariantParquetField.validateShape(field); // must not throw
    }

    public void testValidateShapeRejectsDeviations() {
        Field good = new VariantParquetField().toArrowField("v", false);
        Map<String, String> tag = good.getMetadata();

        Field wrongChildName = new Field(
            "v",
            new FieldType(true, ArrowType.Struct.INSTANCE, null, tag),
            List.of(new Field("meta", FieldType.notNullable(new ArrowType.Binary()), null), good.getChildren().get(1))
        );
        assertThrows(IllegalArgumentException.class, () -> VariantParquetField.validateShape(wrongChildName));

        Field wrongChildType = new Field(
            "v",
            new FieldType(true, ArrowType.Struct.INSTANCE, null, tag),
            List.of(good.getChildren().get(0), new Field("value", FieldType.nullable(new ArrowType.Utf8()), null))
        );
        assertThrows(IllegalArgumentException.class, () -> VariantParquetField.validateShape(wrongChildType));

        Field notStruct = new Field("v", new FieldType(true, new ArrowType.Binary(), null, tag), null);
        assertThrows(IllegalArgumentException.class, () -> VariantParquetField.validateShape(notStruct));

        Field untagged = new Field("v", FieldType.nullable(ArrowType.Struct.INSTANCE), good.getChildren());
        assertThrows(IllegalArgumentException.class, () -> VariantParquetField.validateShape(untagged));
    }

    // ---- encoder ----

    public void testEncoderRoundTripsThroughParquetJavaDecoder() throws IOException {
        VariantBytes bytes = encode(DOC_1);
        Variant v = new Variant(bytes.value(), bytes.metadata());
        assertEquals(Variant.Type.OBJECT, v.getType());
        assertEquals(2, v.numObjectElements());
        assertEquals(42L, v.getFieldByKey("id").getLong());
        Variant user = v.getFieldByKey("user");
        assertEquals(7L, user.getFieldByKey("id").getLong());
        Variant tags = user.getFieldByKey("tags");
        assertEquals(Variant.Type.ARRAY, tags.getType());
        assertEquals(2, tags.numArrayElements());
        assertEquals("x", tags.getElementAtIndex(0).getString());
        assertEquals("yz", tags.getElementAtIndex(1).getString());
        // The encoder narrows 42 to int8.
        assertEquals(Variant.Type.BYTE, v.getFieldByKey("id").getType());
    }

    public void testEncoderHandlesAllScalarTokens() throws IOException {
        Variant v2 = toVariant(encode(DOC_2));
        assertEquals("seven", v2.getFieldByKey("user").getFieldByKey("id").getString());
        assertTrue(v2.getFieldByKey("flag").getBoolean());
        assertEquals(0.5d, v2.getFieldByKey("ratio").getDouble(), 0.0);

        Variant v3 = toVariant(encode(DOC_3));
        Variant deeper = v3.getFieldByKey("nested").getFieldByKey("deep").getFieldByKey("deeper");
        assertEquals(3, deeper.numArrayElements());
        assertEquals(Variant.Type.NULL, deeper.getElementAtIndex(2).getFieldByKey("k").getType());
        assertEquals(Variant.Type.DECIMAL16, v3.getFieldByKey("big").getType());
        assertEquals("123456789012345678901234567890", v3.getFieldByKey("big").getDecimal().toPlainString());
    }

    // ---- write + read back through the native bridge ----

    public void testWriteVariantColumnAndReadBackThroughRust() throws Exception {
        Path file = createTempDir().resolve("poc1-variant.parquet");
        Schema schema = variantSchema();

        NativeParquetWriter writer = new NativeParquetWriter(file.toString());
        try (ArrowExport export = exportSchema(schema)) {
            writer.initialize("poc1", export.getSchemaAddress(), ParquetSortConfig.empty(), 0L);
        }
        List<VariantBytes> docs = new ArrayList<>();
        docs.add(encode(DOC_1));
        docs.add(encode(DOC_2));
        docs.add(null); // missing field -> null variant
        docs.add(encode(DOC_3));
        try (ArrowExport export = exportRows(schema, docs)) {
            writer.write(export.getArrayAddress(), export.getSchemaAddress());
        }
        writer.flush();
        assertEquals(4, writer.getMetadata().numRows());
        assertTrue(Files.exists(file));
        // Run with -Dtests.leaveTemporary=true to keep the file for parquet-schema / DuckDB inspection.

        // Footer: the Rust side must have written a group with the two leaves. parquet-schema (CLI)
        // is used out-of-band to assert the VARIANT logical type annotation; here we assert the leaf
        // paths exist, which only happens when the struct crossed the FFI intact.
        String columns = RustBridge.getColumnMetadata(file.toString());
        assertTrue(columns, columns.contains("\"v.metadata\""));
        assertTrue(columns, columns.contains("\"v.value\""));

        // Read side: the arrow reader re-attaches the extension type from the VARIANT annotation,
        // VariantArray::try_new accepts the shape, variant_to_json decodes each row.
        String json = RustBridge.readAsJson(file.toString());
        assertFalse("variant column was not recognised on read: " + json, json.contains("<unsupported"));
        List<Map<String, Object>> rows = parseRows(json);
        assertEquals(4, rows.size());
        assertEquals(1, rows.get(0).get("id"));
        @SuppressWarnings("unchecked")
        Map<String, Object> v1 = (Map<String, Object>) rows.get(0).get("v");
        assertEquals(42, v1.get("id"));
        @SuppressWarnings("unchecked")
        Map<String, Object> v1user = (Map<String, Object>) v1.get("user");
        assertEquals(7, v1user.get("id"));
        assertEquals(List.of("x", "yz"), v1user.get("tags"));
        @SuppressWarnings("unchecked")
        Map<String, Object> v2 = (Map<String, Object>) rows.get(1).get("v");
        assertEquals(true, v2.get("flag"));
        assertEquals(0.5, v2.get("ratio"));
        assertNull(rows.get(2).get("v"));
        @SuppressWarnings("unchecked")
        Map<String, Object> v3 = (Map<String, Object>) rows.get(3).get("v");
        assertNotNull(v3.get("nested"));
    }

    /**
     * D-17 negative test: a tagged struct with the wrong child shape must be refused at writer
     * creation by the Rust guard, so no VARIANT-annotated but undecodable segment is produced.
     */
    public void testMisShapedTaggedStructIsRejectedAtWriterCreation() throws Exception {
        Path file = createTempDir().resolve("poc1-misshaped.parquet");
        Map<String, String> tag = Map.of(VariantParquetField.EXTENSION_NAME_KEY, VariantParquetField.VARIANT_EXTENSION_NAME);
        Field bad = new Field(
            "v",
            new FieldType(true, ArrowType.Struct.INSTANCE, null, tag),
            List.of(
                new Field("meta", FieldType.notNullable(new ArrowType.Binary()), null),
                new Field("value", FieldType.nullable(new ArrowType.Utf8()), null)
            )
        );
        Schema schema = new Schema(List.of(new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null), bad));

        // Java guard catches it first in production; here we bypass it to prove the Rust guard.
        assertThrows(IllegalArgumentException.class, () -> VariantParquetField.validateShape(bad));

        NativeParquetWriter writer = new NativeParquetWriter(file.toString());
        try (ArrowExport export = exportSchema(schema)) {
            IOException e = expectThrows(
                IOException.class,
                () -> writer.initialize("poc1-bad", export.getSchemaAddress(), ParquetSortConfig.empty(), 0L)
            );
            assertTrue(e.getMessage(), e.getMessage().contains("Variant field 'v'"));
        }
        assertFalse(writer.isInitialized());
        assertFalse("no file may be produced for a rejected schema", Files.exists(file));
    }

    // ---- helpers ----

    private static Schema variantSchema() {
        return new Schema(
            List.of(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                new VariantParquetField().toArrowField("v", false)
            )
        );
    }

    private static VariantBytes encode(String json) throws IOException {
        try (XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, null, json)) {
            parser.nextToken();
            return VariantEncoder.encode(parser);
        }
    }

    private static Variant toVariant(VariantBytes bytes) {
        return new Variant(bytes.value(), bytes.metadata());
    }

    private ArrowExport exportSchema(Schema schema) {
        ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
        Data.exportSchema(allocator, schema, null, arrowSchema);
        return new ArrowExport(null, arrowSchema);
    }

    private ArrowExport exportRows(Schema schema, List<VariantBytes> docs) {
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            StructVector v = (StructVector) root.getVector("v");
            ((VarBinaryVector) v.getChild("metadata")).allocateNew(1024, docs.size());
            ((VarBinaryVector) v.getChild("value")).allocateNew(1024, docs.size());
            for (int i = 0; i < docs.size(); i++) {
                ids.setSafe(i, i + 1);
                VariantParquetField.writeRow(v, i, docs.get(i));
            }
            v.setValueCount(docs.size());
            root.setRowCount(docs.size());
            ArrowArray array = ArrowArray.allocateNew(allocator);
            ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
            Data.exportVectorSchemaRoot(allocator, root, null, array, arrowSchema);
            return new ArrowExport(array, arrowSchema);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> parseRows(String json) throws IOException {
        try (XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, null, json)) {
            return (List<Map<String, Object>>) (List<?>) parser.list();
        }
    }
}
