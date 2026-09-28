/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.fields.core.data.variant;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.parquet.variant.Variant;
import org.apache.parquet.variant.VariantArrayBuilder;
import org.apache.parquet.variant.VariantBuilder;
import org.apache.parquet.variant.VariantObjectBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.core.xcontent.XContentParser.Token;
import org.opensearch.parquet.fields.core.data.variant.VariantParquetField.VariantBytes;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * D-14 bench prototype (test sources only): {@link VariantEncoder} plus encode-time typed
 * extraction of a declared set of dotted paths into Arrow vectors, driven from the same single
 * pass over {@link XContentParser} tokens. The declared paths form a small trie; while an object
 * is walked, each key is looked up in the current trie node, and a key that hits a typed leaf has
 * its scalar written to that leaf's vector at {@code row}. In {@code removeFromResidual} mode the
 * key/value is then omitted from the residual Variant (the shredded {@code value} column); in
 * extract-only mode the residual is byte-identical to the unshredded encoding.
 * <p>
 * Type mismatches (declared INT64 but the document has a string) fall back to the residual and a
 * null typed value, which is the shredding spec's semantics. Arrays reset the trie position (no
 * declared paths cross an array in POC-3's shape).
 */
final class ShreddingVariantEncoder {

    enum LeafType {
        INT64,
        UTF8,
        BOOL
    }

    private static final class Node {
        final Map<String, Node> children = new HashMap<>();
        LeafType leaf;
        int slot = -1;
    }

    private final Node root = new Node();
    private final List<FieldVector> vectors = new ArrayList<>();
    private final List<String> paths = new ArrayList<>();
    private final boolean removeFromResidual;
    private boolean[] seen;

    ShreddingVariantEncoder(boolean removeFromResidual) {
        this.removeFromResidual = removeFromResidual;
    }

    /** Declares {@code dottedPath} as a typed leaf; vectors are created against {@code allocator}. */
    ShreddingVariantEncoder withPath(String dottedPath, LeafType type, BufferAllocator allocator) {
        Node node = root;
        for (String part : dottedPath.split("\\.")) {
            node = node.children.computeIfAbsent(part, k -> new Node());
        }
        if (node.leaf != null || node.children.isEmpty() == false) {
            throw new IllegalArgumentException("path conflicts with an existing declaration: " + dottedPath);
        }
        node.leaf = type;
        node.slot = vectors.size();
        FieldVector v = switch (type) {
            case INT64 -> new BigIntVector(dottedPath, allocator);
            case UTF8 -> new VarCharVector(dottedPath, allocator);
            case BOOL -> new BitVector(dottedPath, allocator);
        };
        vectors.add(v);
        paths.add(dottedPath);
        seen = new boolean[vectors.size()];
        return this;
    }

    List<FieldVector> vectors() {
        return vectors;
    }

    List<String> paths() {
        return paths;
    }

    /** Pre-sizes the typed vectors for {@code rows} rows. */
    void allocate(int rows) {
        for (FieldVector v : vectors) {
            v.setInitialCapacity(rows);
            v.allocateNew();
        }
    }

    /** Resets typed vectors between bench runs (keeps allocations). */
    void reset() {
        for (FieldVector v : vectors) {
            v.reset();
        }
    }

    void setValueCount(int rows) {
        for (FieldVector v : vectors) {
            v.setValueCount(rows);
        }
    }

    void close() {
        for (FieldVector v : vectors) {
            v.close();
        }
    }

    /**
     * Encodes the value at the parser's current token, extracting declared paths into row {@code row}
     * of the typed vectors. Declared leaves absent from the document are written as null.
     */
    VariantBytes encode(XContentParser parser, int row) throws IOException {
        java.util.Arrays.fill(seen, false);
        VariantBuilder builder = new VariantBuilder();
        appendValue(builder, parser, parser.currentToken(), root, row);
        for (int i = 0; i < seen.length; i++) {
            if (seen[i] == false) {
                setNull(vectors.get(i), row);
            }
        }
        return toBytes(builder.build());
    }

    private void appendValue(VariantBuilder builder, XContentParser parser, Token token, Node node, int row) throws IOException {
        switch (token) {
            case START_OBJECT -> {
                VariantObjectBuilder object = builder.startObject();
                Token next;
                while ((next = parser.nextToken()) != Token.END_OBJECT) {
                    if (next != Token.FIELD_NAME) {
                        throw new IllegalStateException("Expected field name inside object, got " + next);
                    }
                    String key = parser.currentName();
                    Token valueToken = parser.nextToken();
                    Node child = node == null ? null : node.children.get(key);
                    if (child != null && child.leaf != null) {
                        if (tryExtract(child, parser, valueToken, row)) {
                            if (removeFromResidual) {
                                continue; // shredded: typed_value carries it, residual omits it
                            }
                            object.appendKey(key);
                            appendValue(object, parser, valueToken, null, row);
                            continue;
                        }
                        // type mismatch: stays in the residual, typed leaf is null (seen[] stays false)
                        object.appendKey(key);
                        appendValue(object, parser, valueToken, null, row);
                        continue;
                    }
                    object.appendKey(key);
                    appendValue(object, parser, valueToken, child, row);
                }
                builder.endObject();
            }
            case START_ARRAY -> {
                VariantArrayBuilder array = builder.startArray();
                Token next;
                while ((next = parser.nextToken()) != Token.END_ARRAY) {
                    appendValue(array, parser, next, null, row);
                }
                builder.endArray();
            }
            case VALUE_STRING -> builder.appendString(parser.text());
            case VALUE_NUMBER -> appendNumber(builder, parser);
            case VALUE_BOOLEAN -> builder.appendBoolean(parser.booleanValue());
            case VALUE_NULL -> builder.appendNull();
            case VALUE_EMBEDDED_OBJECT -> builder.appendBinary(ByteBuffer.wrap(parser.binaryValue()));
            default -> throw new IllegalStateException("Unexpected token in variant value: " + token);
        }
    }

    /** Writes the scalar at the parser position into the leaf's vector; false if the token does not fit the declared type. */
    private boolean tryExtract(Node leaf, XContentParser parser, Token token, int row) throws IOException {
        FieldVector v = vectors.get(leaf.slot);
        switch (leaf.leaf) {
            case INT64 -> {
                if (token != Token.VALUE_NUMBER) {
                    return false;
                }
                XContentParser.NumberType nt = parser.numberType();
                if (nt != XContentParser.NumberType.INT && nt != XContentParser.NumberType.LONG) {
                    return false;
                }
                ((BigIntVector) v).setSafe(row, parser.longValue());
            }
            case UTF8 -> {
                if (token != Token.VALUE_STRING) {
                    return false;
                }
                ((VarCharVector) v).setSafe(row, parser.text().getBytes(StandardCharsets.UTF_8));
            }
            case BOOL -> {
                if (token != Token.VALUE_BOOLEAN) {
                    return false;
                }
                ((BitVector) v).setSafe(row, parser.booleanValue() ? 1 : 0);
            }
        }
        seen[leaf.slot] = true;
        return true;
    }

    private static void setNull(FieldVector v, int row) {
        switch (v) {
            case BigIntVector b -> b.setNull(row);
            case VarCharVector s -> s.setNull(row);
            case BitVector bit -> bit.setNull(row);
            default -> throw new IllegalStateException(v.getClass().getName());
        }
    }

    private static void appendNumber(VariantBuilder builder, XContentParser parser) throws IOException {
        switch (parser.numberType()) {
            case INT, LONG -> appendNarrowedLong(builder, parser.longValue());
            case FLOAT -> builder.appendFloat(parser.floatValue());
            case DOUBLE -> builder.appendDouble(parser.doubleValue());
            case BIG_INTEGER, BIG_DECIMAL -> builder.appendDecimal(new BigDecimal(parser.text()));
        }
    }

    private static void appendNarrowedLong(VariantBuilder builder, long v) {
        if (v >= Byte.MIN_VALUE && v <= Byte.MAX_VALUE) {
            builder.appendByte((byte) v);
        } else if (v >= Short.MIN_VALUE && v <= Short.MAX_VALUE) {
            builder.appendShort((short) v);
        } else if (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) {
            builder.appendInt((int) v);
        } else {
            builder.appendLong(v);
        }
    }

    private static VariantBytes toBytes(Variant variant) {
        return new VariantBytes(copy(variant.getMetadataBuffer()), copy(variant.getValueBuffer()));
    }

    private static byte[] copy(ByteBuffer buffer) {
        ByteBuffer slice = buffer.slice();
        byte[] out = new byte[slice.remaining()];
        slice.get(out);
        return out;
    }
}
