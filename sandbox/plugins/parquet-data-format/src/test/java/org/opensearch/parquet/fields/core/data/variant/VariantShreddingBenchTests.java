/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.fields.core.data.variant;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.parquet.variant.Variant;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.parquet.fields.core.data.variant.VariantParquetField.VariantBytes;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;

/**
 * D-14 bench: JVM typed extraction cost at encode time, on the POC-3 CloudTrail corpus and the
 * POC-3 six-path shredding shape, so the number is directly comparable to the Rust smoke
 * ({@code json_to_variant} 26.3 us/row, {@code shred_variant} 19.1 us/row).
 * <p>
 * Measures per document, single pass over {@link XContentParser} tokens (parser creation included
 * in every mode, as it is in production):
 * <ol>
 *   <li>unshredded {@link VariantEncoder} (baseline)</li>
 *   <li>baseline + typed extraction of the 6 paths, residual unchanged (extract-only)</li>
 *   <li>baseline + typed extraction of the 6 paths, paths removed from the residual (shredded value column)</li>
 * </ol>
 * Warm-up {@value #WARMUP} docs per mode, then median of {@value #RUNS} runs over the full corpus.
 * Run with {@code -Dtests.output=true -Dtests.heap.size=2g}; rows via {@code -Dtests.variant.bench.rows}.
 */
public class VariantShreddingBenchTests extends OpenSearchTestCase {

    private static final int RUNS = 5;
    private static final int WARMUP = 20_000;

    private BufferAllocator allocator;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        allocator = new RootAllocator();
    }

    @Override
    public void tearDown() throws Exception {
        allocator.close();
        super.tearDown();
    }

    private static ShreddingVariantEncoder poc3Shape(boolean remove, BufferAllocator allocator) {
        return new ShreddingVariantEncoder(remove).withPath(
            "responseElements.statusCode",
            ShreddingVariantEncoder.LeafType.INT64,
            allocator
        )
            .withPath("eventName", ShreddingVariantEncoder.LeafType.UTF8, allocator)
            .withPath("eventTime", ShreddingVariantEncoder.LeafType.UTF8, allocator)
            .withPath("awsRegion", ShreddingVariantEncoder.LeafType.UTF8, allocator)
            .withPath("readOnly", ShreddingVariantEncoder.LeafType.BOOL, allocator)
            .withPath("userIdentity.accountId", ShreddingVariantEncoder.LeafType.UTF8, allocator);
    }

    // ---- correctness of the prototype (cheap, always runs) ----

    public void testCorpusMatchesRustSmokeShape() {
        String[] docs = CloudTrailCorpus.corpus(2000);
        long total = 0;
        for (String d : docs) {
            total += d.getBytes(StandardCharsets.UTF_8).length;
        }
        // Rust smoke on 100k docs reported avg 1564 B; the first 2k should land in the same band.
        long avg = total / docs.length;
        assertTrue("avg doc length " + avg, avg > 1500 && avg < 1620);
        assertTrue(
            docs[0].startsWith("{\"eventVersion\":\"1.08\",\"eventTime\":\"2026-09-24T00:00:00Z\",\"eventSource\":\"s3.amazonaws.com\"")
        );
        assertTrue(docs[0].contains("\"responseElements\":{\"statusCode\":"));
    }

    public void testExtractedValuesMatchResidualAndUnshredded() throws IOException {
        String[] docs = CloudTrailCorpus.corpus(64);
        ShreddingVariantEncoder keep = poc3Shape(false, allocator);
        ShreddingVariantEncoder remove = poc3Shape(true, allocator);
        try {
            keep.allocate(docs.length);
            remove.allocate(docs.length);
            for (int i = 0; i < docs.length; i++) {
                byte[] bytes = docs[i].getBytes(StandardCharsets.UTF_8);
                VariantBytes base = encodeBaseline(bytes);
                VariantBytes kept = encodeWith(keep, bytes, i);
                VariantBytes removed = encodeWith(remove, bytes, i);
                // extract-only leaves the residual byte-identical to the unshredded encoding
                assertArrayEquals(base.metadata(), kept.metadata());
                assertArrayEquals(base.value(), kept.value());
                // shredded residual is strictly smaller and no longer contains the declared leaves
                assertTrue(removed.value().length < base.value().length);
                Variant full = new Variant(base.value(), base.metadata());
                Variant residual = new Variant(removed.value(), removed.metadata());
                assertNull(residual.getFieldByKey("eventName"));
                assertNull(residual.getFieldByKey("responseElements").getFieldByKey("statusCode"));
                assertNotNull(residual.getFieldByKey("responseElements").getFieldByKey("x-amz-request-id"));
                assertNull(residual.getFieldByKey("userIdentity").getFieldByKey("accountId"));
                // typed leaves equal the values decoded from the unshredded Variant
                assertEquals(full.getFieldByKey("responseElements").getFieldByKey("statusCode").getLong(), longAt(remove, 0, i));
                assertEquals(full.getFieldByKey("eventName").getString(), stringAt(remove, 1, i));
                assertEquals(full.getFieldByKey("eventTime").getString(), stringAt(remove, 2, i));
                assertEquals(full.getFieldByKey("awsRegion").getString(), stringAt(remove, 3, i));
                assertEquals(full.getFieldByKey("readOnly").getBoolean(), ((BitVector) remove.vectors().get(4)).get(i) == 1);
                assertEquals(full.getFieldByKey("userIdentity").getFieldByKey("accountId").getString(), stringAt(remove, 5, i));
            }
            // a declared leaf with the wrong type stays in the residual and reads back null
            byte[] mismatch = "{\"eventName\":7,\"responseElements\":{\"statusCode\":\"200\"}}".getBytes(StandardCharsets.UTF_8);
            remove.reset();
            VariantBytes r = encodeWith(remove, mismatch, 0);
            Variant residual = new Variant(r.value(), r.metadata());
            assertEquals(7L, residual.getFieldByKey("eventName").getLong());
            assertEquals("200", residual.getFieldByKey("responseElements").getFieldByKey("statusCode").getString());
            assertTrue(remove.vectors().get(0).isNull(0));
            assertTrue(remove.vectors().get(1).isNull(0));
            assertTrue(remove.vectors().get(2).isNull(0));
        } finally {
            keep.close();
            remove.close();
        }
    }

    // ---- the bench ----

    public void testEncodeTimeExtractionCost() throws IOException {
        int rows = Integer.getInteger("tests.variant.bench.rows", 100_000);
        String[] strings = CloudTrailCorpus.corpus(rows);
        byte[][] docs = new byte[rows][];
        long totalBytes = 0;
        for (int i = 0; i < rows; i++) {
            docs[i] = strings[i].getBytes(StandardCharsets.UTF_8);
            totalBytes += docs[i].length;
        }
        strings = null;
        int warm = Math.min(WARMUP, rows);
        StringBuilder report = new StringBuilder();
        report.append(
            String.format(
                Locale.ROOT,
                "%n=== D-14 JVM encode-time typed extraction: %d CloudTrail-like docs (avg %d B), warm-up %d, median of %d, asserts=%s ===%n",
                rows,
                totalBytes / rows,
                warm,
                RUNS,
                VariantShreddingBenchTests.class.desiredAssertionStatus()
            )
        );

        ShreddingVariantEncoder keep = poc3Shape(false, allocator);
        ShreddingVariantEncoder remove = poc3Shape(true, allocator);
        try {
            keep.allocate(rows);
            remove.allocate(rows);

            // warm-up all three paths
            long sink = 0;
            for (int i = 0; i < warm; i++) {
                sink += encodeBaseline(docs[i]).value().length;
                sink += encodeWith(keep, docs[i], i).value().length;
                sink += encodeWith(remove, docs[i], i).value().length;
            }

            double[] base = new double[RUNS];
            double[] extract = new double[RUNS];
            double[] shred = new double[RUNS];
            long[] residualBytes = new long[3];
            for (int r = 0; r < RUNS; r++) {
                long t0 = System.nanoTime();
                long b = 0;
                for (int i = 0; i < rows; i++) {
                    b += encodeBaseline(docs[i]).value().length;
                }
                base[r] = (System.nanoTime() - t0) / 1000.0 / rows;
                residualBytes[0] = b;

                keep.reset();
                t0 = System.nanoTime();
                b = 0;
                for (int i = 0; i < rows; i++) {
                    b += encodeWith(keep, docs[i], i).value().length;
                }
                keep.setValueCount(rows);
                extract[r] = (System.nanoTime() - t0) / 1000.0 / rows;
                residualBytes[1] = b;

                remove.reset();
                t0 = System.nanoTime();
                b = 0;
                for (int i = 0; i < rows; i++) {
                    b += encodeWith(remove, docs[i], i).value().length;
                }
                remove.setValueCount(rows);
                shred[r] = (System.nanoTime() - t0) / 1000.0 / rows;
                residualBytes[2] = b;
                sink += b;
            }
            double mBase = median(base), mExtract = median(extract), mShred = median(shred);
            report.append(String.format(Locale.ROOT, "samples baseline   us/doc: %s%n", fmt(base)));
            report.append(String.format(Locale.ROOT, "samples +extract   us/doc: %s%n", fmt(extract)));
            report.append(String.format(Locale.ROOT, "samples +shred     us/doc: %s%n", fmt(shred)));
            report.append(String.format(Locale.ROOT, "%-52s %10s %10s %8s%n", "mode", "us/doc", "delta", "delta%"));
            report.append(String.format(Locale.ROOT, "%-52s %10.2f %10s %8s%n", "(i)  unshredded VariantEncoder", mBase, "-", "-"));
            report.append(
                String.format(
                    Locale.ROOT,
                    "%-52s %10.2f %10.2f %7.1f%%%n",
                    "(ii) encoder + typed extraction (6 paths), residual kept",
                    mExtract,
                    mExtract - mBase,
                    100.0 * (mExtract - mBase) / mBase
                )
            );
            report.append(
                String.format(
                    Locale.ROOT,
                    "%-52s %10.2f %10.2f %7.1f%%%n",
                    "(iii) encoder + typed extraction, paths removed from residual",
                    mShred,
                    mShred - mBase,
                    100.0 * (mShred - mBase) / mBase
                )
            );
            report.append(
                String.format(
                    Locale.ROOT,
                    "residual value bytes/doc: unshredded %.0f, extract-only %.0f, shredded %.0f%n",
                    (double) residualBytes[0] / rows,
                    (double) residualBytes[1] / rows,
                    (double) residualBytes[2] / rows
                )
            );
            report.append("sink=").append(sink).append('\n');
            System.out.println(report);
            logger.info(report.toString());
            assertTrue(sink != 0);
        } finally {
            keep.close();
            remove.close();
        }
    }

    // ---- helpers ----

    private static VariantBytes encodeBaseline(byte[] doc) throws IOException {
        try (XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, null, doc)) {
            parser.nextToken();
            return VariantEncoder.encode(parser);
        }
    }

    private static VariantBytes encodeWith(ShreddingVariantEncoder encoder, byte[] doc, int row) throws IOException {
        try (XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, null, doc)) {
            parser.nextToken();
            return encoder.encode(parser, row);
        }
    }

    private static long longAt(ShreddingVariantEncoder e, int slot, int row) {
        return ((BigIntVector) e.vectors().get(slot)).get(row);
    }

    private static String stringAt(ShreddingVariantEncoder e, int slot, int row) {
        return new String(((VarCharVector) e.vectors().get(slot)).get(row), StandardCharsets.UTF_8);
    }

    private static double median(double[] xs) {
        double[] s = xs.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    private static String fmt(double[] xs) {
        StringBuilder b = new StringBuilder();
        for (double x : xs) {
            b.append(String.format(Locale.ROOT, "%.2f ", x));
        }
        return b.toString().trim();
    }
}
