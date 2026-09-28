/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.parquet.fields.core.data.variant;

import java.util.Locale;

/**
 * Faithful Java port of the CloudTrail-like corpus generator in
 * {@code sandbox/plugins/analytics-backend-datafusion/rust/src/indexed_table/tests_e2e/variant_poc3_smoke.rs}
 * ({@code Lcg}, {@code cloudtrail_doc}, {@code corpus}) so JVM and Rust D-14 numbers are measured on
 * byte-identical documents. Argument evaluation order (including the short-circuited
 * {@code readOnly} draw) mirrors the Rust {@code format!} call exactly.
 */
final class CloudTrailCorpus {

    private CloudTrailCorpus() {}

    /** {@code Lcg(0x5eed)} in the Rust smoke. */
    static final long SEED = 0x5eed;

    /** 64-bit LCG, same constants as the Rust smoke; {@code >> 33} keeps 31 bits so every draw is non-negative. */
    static final class Lcg {
        private long state;

        Lcg(long seed) {
            this.state = seed;
        }

        long next() {
            state = state * 6364136223846793005L + 1442695040888963407L;
            return state >>> 33;
        }

        String pick(String[] xs) {
            return xs[(int) (next() % xs.length)];
        }
    }

    private static final String[] EVENTS = {
        "GetObject",
        "PutObject",
        "ListObjects",
        "DeleteObject",
        "HeadObject",
        "CopyObject",
        "GetBucketAcl",
        "PutObjectTagging" };
    private static final String[] REGIONS = { "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2" };
    private static final String[] TEAMS = { "search", "indexing", "storage", "platform", "security" };
    private static final String[] ENVS = { "prod", "beta", "gamma" };

    static String cloudtrailDoc(int i, Lcg rng) {
        long statusDraw = rng.next() % 100;
        int status = statusDraw <= 89 ? 200 : statusDraw <= 95 ? 403 : statusDraw <= 98 ? 404 : 500;
        long acct = 100_000_000_000L + (rng.next() % 900_000_000_000L);
        long bucket = rng.next() % 5000;
        long obj = rng.next() % 1_000_000;

        // Arguments in the exact order Rust evaluates them for format!.
        int hh = (i / 3600) % 24, mm = (i / 60) % 60, ss = i % 60;
        String eventName = rng.pick(EVENTS);
        String region = rng.pick(REGIONS);
        long ip1 = rng.next() % 256, ip2 = rng.next() % 256, ip3 = rng.next() % 256, cli = rng.next() % 60;
        long principal = rng.next();
        int sess = i % 1000;
        String mfa = rng.next() % 3 == 0 ? "true" : "false";
        long issuerPrincipal = rng.next();
        String team = rng.pick(TEAMS);
        String env = rng.pick(ENVS);
        long reqId = rng.next(), id2a = rng.next(), id2b = rng.next();
        long requestID = rng.next();
        long e1 = rng.next() & 0xffff_ffffL;   // as u32
        long e2 = rng.next() & 0xffff;         // as u16
        long e3 = rng.next() & 0xffff;
        long e4 = rng.next() & 0xffff;
        long e5 = rng.next() & 0xffff_ffff_ffffL;
        String readOnly = (status == 200 && rng.next() % 2 == 0) ? "true" : "false"; // short-circuit preserved

        StringBuilder b = new StringBuilder(1700);
        b.append("{\"eventVersion\":\"1.08\",\"eventTime\":\"2026-09-24T")
            .append(pad2(hh))
            .append(':')
            .append(pad2(mm))
            .append(':')
            .append(pad2(ss))
            .append("Z\",");
        b.append("\"eventSource\":\"s3.amazonaws.com\",\"eventName\":\"")
            .append(eventName)
            .append("\",\"awsRegion\":\"")
            .append(region)
            .append("\",");
        b.append("\"sourceIPAddress\":\"10.")
            .append(ip1)
            .append('.')
            .append(ip2)
            .append('.')
            .append(ip3)
            .append("\",\"userAgent\":\"aws-cli/2.17.")
            .append(cli)
            .append(" Python/3.12 Linux/5.10 exe/x86_64\",");
        b.append("\"userIdentity\":{\"type\":\"AssumedRole\",\"principalId\":\"AROA")
            .append(hex16Upper(principal))
            .append(":session-")
            .append(sess)
            .append("\",");
        b.append("\"arn\":\"arn:aws:sts::")
            .append(acct)
            .append(":assumed-role/DataRole/session-")
            .append(sess)
            .append("\",\"accountId\":\"")
            .append(acct)
            .append("\",");
        b.append("\"sessionContext\":{\"attributes\":{\"mfaAuthenticated\":\"")
            .append(mfa)
            .append("\",\"creationDate\":\"2026-09-24T09:00:00Z\"},");
        b.append("\"sessionIssuer\":{\"type\":\"Role\",\"principalId\":\"AROA")
            .append(hex16Upper(issuerPrincipal))
            .append("\",\"arn\":\"arn:aws:iam::")
            .append(acct)
            .append(":role/DataRole\",\"accountId\":\"")
            .append(acct)
            .append("\",\"userName\":\"DataRole\"}}},");
        b.append("\"requestParameters\":{\"bucketName\":\"bucket-")
            .append(bucket)
            .append("\",\"key\":\"data/dt=2026-09-24/part-")
            .append(pad7(obj))
            .append(".parquet\",");
        b.append("\"Host\":\"bucket-").append(bucket).append(".s3.amazonaws.com\",\"x-amz-acl\":\"private\",");
        b.append("\"tags\":[{\"key\":\"team\",\"value\":\"")
            .append(team)
            .append("\"},{\"key\":\"env\",\"value\":\"")
            .append(env)
            .append("\"}]},");
        b.append("\"responseElements\":{\"statusCode\":")
            .append(status)
            .append(",\"x-amz-request-id\":\"")
            .append(hex16Upper(reqId))
            .append("\",\"x-amz-id-2\":\"")
            .append(hex16Upper(id2a))
            .append(hex16Upper(id2b))
            .append("\"},");
        b.append("\"requestID\":\"")
            .append(hex16Upper(requestID))
            .append("\",\"eventID\":\"")
            .append(hexLower(e1, 8))
            .append('-')
            .append(hexLower(e2, 4))
            .append('-')
            .append(hexLower(e3, 4))
            .append('-')
            .append(hexLower(e4, 4))
            .append('-')
            .append(hexLower(e5, 12))
            .append("\",\"readOnly\":")
            .append(readOnly)
            .append(',');
        b.append("\"resources\":[{\"type\":\"AWS::S3::Object\",\"ARN\":\"arn:aws:s3:::bucket-")
            .append(bucket)
            .append("/data/part-")
            .append(pad7(obj))
            .append(".parquet\"},");
        b.append("{\"type\":\"AWS::S3::Bucket\",\"ARN\":\"arn:aws:s3:::bucket-")
            .append(bucket)
            .append("\",\"accountId\":\"")
            .append(acct)
            .append("\"}],");
        b.append("\"eventType\":\"AwsApiCall\",\"managementEvent\":false,\"recipientAccountId\":\"")
            .append(acct)
            .append("\",\"eventCategory\":\"Data\",");
        b.append(
            "\"tlsDetails\":{\"tlsVersion\":\"TLSv1.3\",\"cipherSuite\":\"TLS_AES_128_GCM_SHA256\",\"clientProvidedHostHeader\":\"bucket-"
        ).append(bucket).append(".s3.amazonaws.com\"}}");
        return b.toString();
    }

    /** Same as Rust {@code corpus(n)}: one LCG seeded {@code 0x5eed}, docs generated in order. */
    static String[] corpus(int n) {
        Lcg rng = new Lcg(SEED);
        String[] docs = new String[n];
        for (int i = 0; i < n; i++) {
            docs[i] = cloudtrailDoc(i, rng);
        }
        return docs;
    }

    private static String pad2(int v) {
        return v < 10 ? "0" + v : Integer.toString(v);
    }

    private static String pad7(long v) {
        return String.format(Locale.ROOT, "%07d", v);
    }

    private static String hex16Upper(long v) {
        return String.format(Locale.ROOT, "%016X", v);
    }

    private static String hexLower(long v, int width) {
        return String.format(Locale.ROOT, "%0" + width + "x", v);
    }
}
