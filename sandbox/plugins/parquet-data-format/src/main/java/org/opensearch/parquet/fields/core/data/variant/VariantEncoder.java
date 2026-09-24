/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.fields.core.data.variant;

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

/**
 * Encodes the subtree under an {@link XContentParser} position into the Parquet Variant binary
 * encoding by driving parquet-java's streaming {@link VariantBuilder} directly from the token
 * stream. The document is parsed once; no intermediate JSON string is materialised.
 * <p>
 * Type mapping is the natural one: objects and arrays nest, strings stay strings, integers are
 * narrowed to the smallest Variant int width that holds them, floats and doubles
 * keep their width, big integers and big decimals become Variant decimals, booleans and nulls
 * map directly, and embedded binary becomes Variant binary. XContent has no temporal token, so
 * dates arrive as strings; typed temporal encoding is a mapping-driven (shredding) concern.
 */
public final class VariantEncoder {

    private VariantEncoder() {}

    /**
     * Encodes the value at the parser's current token. The parser must be positioned on the
     * first token of the value (for an object, {@link Token#START_OBJECT}); on return it is
     * positioned on the last token of that value.
     *
     * @param parser the parser
     * @return the encoded metadata and value blobs
     * @throws IOException on parse failure
     */
    public static VariantBytes encode(XContentParser parser) throws IOException {
        VariantBuilder builder = new VariantBuilder();
        appendValue(builder, parser, parser.currentToken());
        return toBytes(builder.build());
    }

    private static void appendValue(VariantBuilder builder, XContentParser parser, Token token) throws IOException {
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
                    object.appendKey(key);
                    appendValue(object, parser, valueToken);
                }
                builder.endObject();
            }
            case START_ARRAY -> {
                VariantArrayBuilder array = builder.startArray();
                Token next;
                while ((next = parser.nextToken()) != Token.END_ARRAY) {
                    appendValue(array, parser, next);
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

    private static void appendNumber(VariantBuilder builder, XContentParser parser) throws IOException {
        switch (parser.numberType()) {
            case INT, LONG -> appendNarrowedLong(builder, parser.longValue());
            case FLOAT -> builder.appendFloat(parser.floatValue());
            case DOUBLE -> builder.appendDouble(parser.doubleValue());
            case BIG_INTEGER, BIG_DECIMAL -> builder.appendDecimal(new BigDecimal(parser.text()));
        }
    }

    /**
     * {@link VariantBuilder#appendLong} always writes int64; the spec allows int8/16/32 and
     * readers widen transparently, so pick the smallest width that holds the value.
     */
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
