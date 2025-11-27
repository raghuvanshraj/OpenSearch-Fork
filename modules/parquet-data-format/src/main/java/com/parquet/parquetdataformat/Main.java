/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package com.parquet.parquetdataformat;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

public class Main {
    public static void main(String[] args) {
        RootAllocator allocator = new RootAllocator(Integer.MAX_VALUE);
        BufferAllocator childAllocator1 = allocator.newChildAllocator("child-1", 0, Long.MAX_VALUE);
        System.out.println(allocator.getChildAllocators());
        BufferAllocator childAllocator2 = allocator.newChildAllocator("child-2", 0, Long.MAX_VALUE);
        System.out.println(allocator.getChildAllocators());
        childAllocator1.close();
        System.out.println(allocator.getChildAllocators());
        childAllocator2.close();
        System.out.println(allocator.getChildAllocators());
        allocator.close();
    }
}
