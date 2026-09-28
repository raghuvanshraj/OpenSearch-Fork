/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.vsr;

import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.lucene.search.Query;
import org.opensearch.Version;
import org.opensearch.arrow.allocator.ArrowNativeAllocator;
import org.opensearch.arrow.spi.NativeAllocatorPoolConfig;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.engine.dataformat.DocumentInput;
import org.opensearch.index.mapper.KeywordFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.mapper.TextSearchInfo;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.parquet.ParquetBaseTests;
import org.opensearch.parquet.ParquetDataFormatPlugin;
import org.opensearch.parquet.bridge.ParquetFileMetadata;
import org.opensearch.parquet.bridge.RustBridge;
import org.opensearch.parquet.fields.core.data.variant.VariantEncoder;
import org.opensearch.parquet.fields.core.data.variant.VariantParquetField;
import org.opensearch.parquet.fields.core.data.variant.VariantParquetField.VariantBytes;
import org.opensearch.parquet.memory.ArrowBufferPool;
import org.opensearch.parquet.writer.ParquetDocumentInput;
import org.opensearch.search.lookup.SearchLookup;
import org.opensearch.threadpool.FixedExecutorBuilder;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * V-14: the Arrow {@link Field} metadata carrying {@code ARROW:extension:name=arrow.parquet.variant}
 * must survive the production VSR lifecycle on Arrow Java 18.1.0 -- {@link VSRManager#reconcileSchema}
 * (the dynamic-mapping case, which rebuilds the {@code VectorSchemaRoot}) followed by a real
 * {@link VSRManager#maybeRotateActiveVSR} (which creates a fresh {@link ManagedVSR} from the pool's
 * updated schema) -- and the Parquet footer written from the post-rotation VSR must still carry the
 * {@code VARIANT} logical type on the group.
 * <p>
 * Footer proof in-test: {@code parquet_read_as_json} only decodes a struct as Variant when
 * {@code field.try_extension_type::<VariantType>()} succeeds, and the arrow-rs reader only re-attaches
 * that extension from the footer's {@code VARIANT} logical type. So a decoded {@code v} object for a
 * row that was written from the post-rotation VSR is proof the annotation is in the footer.
 * {@code parquet-schema} is run out-of-band (with {@code -Dtests.leaveTemporary=true}) for the
 * literal {@code VARIANT} line.
 */
public class VariantVsrRotationTests extends ParquetBaseTests {

    private static final String DOC_1 = "{\"user\":{\"id\":7,\"tags\":[\"x\",\"yz\"]},\"id\":42}";
    private static final String DOC_2 = "{\"user\":{\"id\":\"seven\",\"name\":\"Ada\"},\"id\":43,\"flag\":true,\"ratio\":0.5}";
    private static final String DOC_3 = "{\"id\":44,\"nested\":{\"deep\":{\"deeper\":[1,2,{\"k\":null}]}}}";

    /** Stand-in for the (not yet written) server-side variant mapper: any MappedFieldType whose typeName is "variant". */
    private static final class VariantFieldType extends MappedFieldType {
        VariantFieldType(String name) {
            super(name, false, false, false, TextSearchInfo.NONE, Map.of());
        }

        @Override
        public ValueFetcher valueFetcher(QueryShardContext context, SearchLookup searchLookup, String format) {
            return null;
        }

        @Override
        public String typeName() {
            return VariantParquetField.CONTENT_TYPE;
        }

        @Override
        public Query termQuery(Object value, QueryShardContext context) {
            return null;
        }
    }

    private ArrowNativeAllocator nativeAllocator;
    private ArrowBufferPool bufferPool;
    private ThreadPool threadPool;
    private IndexSettings indexSettings;
    private BufferAllocator importAllocator;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        RustBridge.initLogger();
        nativeAllocator = new ArrowNativeAllocator();
        nativeAllocator.getOrCreatePool(NativeAllocatorPoolConfig.POOL_INGEST, 0L, Long.MAX_VALUE, null);
        bufferPool = new ArrowBufferPool(Settings.EMPTY, nativeAllocator);
        importAllocator = new RootAllocator();
        Settings idx = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .build();
        indexSettings = new IndexSettings(IndexMetadata.builder("variant-rotation").settings(idx).build(), Settings.EMPTY);
        Settings settings = Settings.builder().put("node.name", "variant-rotation-test").build();
        threadPool = new ThreadPool(
            settings,
            new FixedExecutorBuilder(
                settings,
                ParquetDataFormatPlugin.PARQUET_THREAD_POOL_NAME,
                1,
                -1,
                "thread_pool." + ParquetDataFormatPlugin.PARQUET_THREAD_POOL_NAME
            )
        );
    }

    @Override
    public void tearDown() throws Exception {
        terminate(threadPool);
        importAllocator.close();
        bufferPool.close();
        nativeAllocator.close();
        super.tearDown();
    }

    public void testExtensionMetadataSurvivesReconcileAndRotationAndFooterIsVariant() throws Exception {
        Path file = createTempDir().resolve("v14-rotation.parquet");

        // Initial schema: metadata fields + id + the variant column.
        List<Field> initialFields = new ArrayList<>(metadataFields());
        initialFields.add(new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null));
        initialFields.add(new VariantParquetField().toArrowField("v", false));
        Schema initial = new Schema(initialFields);

        // maxRowsPerVSR=2 forces a rotation on the third document; runAsync=false keeps the
        // frozen-VSR write on the calling thread so the sequence is deterministic.
        VSRManager manager = new VSRManager(file.toString(), indexSettings, initial, bufferPool, 2, threadPool, false, 1L);
        try {
            ManagedVSR vsr1 = manager.getActiveManagedVSR();
            assertVariantTagged("initial VSR schema", vsr1.getSchema().findField("v"));
            assertVariantTagged("initial VSR vector.getField()", ((StructVector) vsr1.getVector("v")).getField());

            // Dynamic mapping: a new, unrelated keyword field appears. reconcileSchema goes through
            // ManagedVSR.addFieldVector, which rebuilds the VectorSchemaRoot from the live vectors
            // and the schema's Field objects (VSRManager.java reconcileSchema -> ManagedVSR.addFieldVector).
            List<Field> reconciled = new ArrayList<>(initialFields);
            reconciled.add(new Field("tag", FieldType.nullable(new ArrowType.Utf8()), null));
            assertTrue(manager.reconcileSchema(new Schema(reconciled)));
            assertSame("reconcile must not swap the active VSR", vsr1, manager.getActiveManagedVSR());
            assertNotNull(vsr1.getSchema().findField("tag"));
            assertVariantTagged("post-reconcile VSR schema", vsr1.getSchema().findField("v"));
            assertVariantTagged("post-reconcile vector.getField()", ((StructVector) vsr1.getVector("v")).getField());

            MappedFieldType idField = new NumberFieldMapper.NumberFieldType("id", NumberFieldMapper.NumberType.INTEGER);
            MappedFieldType tagField = new KeywordFieldMapper.KeywordFieldType("tag");
            MappedFieldType vField = new VariantFieldType("v");
            assignTestCapabilities(idField, ParquetDataFormatPlugin.PARQUET_DATA_FORMAT);
            assignTestCapabilities(tagField, ParquetDataFormatPlugin.PARQUET_DATA_FORMAT);
            assignTestCapabilities(vField, ParquetDataFormatPlugin.PARQUET_DATA_FORMAT);
            assertFalse("variant content type must be registered on the parquet format", vField.getCapabilityMap().isEmpty());

            // Rows 0,1 land in VSR1.
            manager.addDocument(doc(0, idField, 1, vField, encode(DOC_1), tagField, "a"));
            manager.addDocument(doc(1, idField, 2, vField, null, tagField, "b"));
            assertSame(vsr1, manager.getActiveManagedVSR());
            assertEquals(2, vsr1.getRowCount());

            // Row 2 trips the threshold: VSRManager.maybeRotateActiveVSR freezes VSR1, initializes the
            // native writer from VSR1's exported schema, writes it, and installs VSR2 built by
            // VSRPool.createNewVSR from the pool schema updated in reconcileSchema.
            manager.addDocument(doc(2, idField, 3, vField, encode(DOC_2), tagField, "c"));
            ManagedVSR vsr2 = manager.getActiveManagedVSR();
            assertNotSame("expected a VSR rotation on the third document", vsr1, vsr2);
            assertEquals(1, vsr2.getRowCount());

            // V-14 core assertions on the post-rotation VSR.
            Field rotatedSchemaField = vsr2.getSchema().findField("v");
            assertVariantTagged("post-rotation VSR schema", rotatedSchemaField);
            assertVariantTagged("post-rotation vector.getField()", ((StructVector) vsr2.getVector("v")).getField());
            assertNotNull("reconciled field must propagate to the rotated VSR", vsr2.getSchema().findField("tag"));
            assertEquals(
                "extension metadata map must be carried verbatim",
                Map.of(
                    VariantParquetField.EXTENSION_NAME_KEY,
                    VariantParquetField.VARIANT_EXTENSION_NAME,
                    VariantParquetField.EXTENSION_METADATA_KEY,
                    ""
                ),
                rotatedSchemaField.getMetadata()
            );
            // ...and across the C Data Interface as the native writer receives it (importSchema consumes the handle).
            Schema imported = Data.importSchema(importAllocator, vsr2.exportSchema(), null);
            assertVariantTagged("post-rotation C-Data export", imported.findField("v"));

            // Row 3 also in VSR2; flush writes VSR2 through the writer initialized from VSR1's schema.
            manager.addDocument(doc(3, idField, 4, vField, encode(DOC_3), tagField, "d"));
            ParquetFileMetadata metadata = manager.flush();
            assertNotNull(metadata);
            assertEquals(4, metadata.numRows());
            assertTrue(Files.exists(file));

            // Footer: leaf paths for the group exist, and the arrow-rs reader recognises the group as
            // VARIANT for rows written from both VSR1 (rows 0,1) and VSR2 (rows 2,3).
            String columns = RustBridge.getColumnMetadata(file.toString());
            assertTrue(columns, columns.contains("\"v.metadata\""));
            assertTrue(columns, columns.contains("\"v.value\""));
            assertTrue(columns, columns.contains("\"tag\""));

            String json = RustBridge.readAsJson(file.toString());
            List<Map<String, Object>> rows = parseRows(json);
            assertEquals(4, rows.size());
            assertVariantDecoded(rows.get(0), 42);
            assertNull("row 1 had no variant value", rows.get(1).get("v"));
            assertVariantDecoded(rows.get(2), 43); // first row written from the post-rotation VSR
            assertVariantDecoded(rows.get(3), 44);
            assertEquals("c", rows.get(2).get("tag"));
        } finally {
            manager.close();
        }
    }

    // ---- helpers ----

    private static void assertVariantTagged(String where, Field field) {
        assertNotNull(where + ": field 'v' missing", field);
        assertEquals(
            where + ": ARROW:extension:name",
            VariantParquetField.VARIANT_EXTENSION_NAME,
            field.getMetadata().get(VariantParquetField.EXTENSION_NAME_KEY)
        );
        assertEquals(where + ": ARROW:extension:metadata", "", field.getMetadata().get(VariantParquetField.EXTENSION_METADATA_KEY));
        VariantParquetField.validateShape(field);
    }

    @SuppressWarnings("unchecked")
    private static void assertVariantDecoded(Map<String, Object> row, int expectedId) {
        Object v = row.get("v");
        assertTrue("footer did not carry VARIANT for this row group: v=" + v, v instanceof Map);
        assertEquals(expectedId, ((Map<String, Object>) v).get("id"));
    }

    private ParquetDocumentInput doc(
        long rowId,
        MappedFieldType idField,
        int id,
        MappedFieldType vField,
        VariantBytes v,
        MappedFieldType tagField,
        String tag
    ) {
        ParquetDocumentInput doc = new ParquetDocumentInput();
        populateMetadataFields(doc);
        doc.setRowId(DocumentInput.ROW_ID_FIELD, rowId);
        doc.addField(idField, id);
        doc.addField(vField, v);
        doc.addField(tagField, tag);
        return doc;
    }

    private static VariantBytes encode(String json) throws IOException {
        try (XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, null, json)) {
            parser.nextToken();
            return VariantEncoder.encode(parser);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> parseRows(String json) throws IOException {
        try (XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, null, json)) {
            return (List<Map<String, Object>>) (List<?>) parser.list();
        }
    }
}
