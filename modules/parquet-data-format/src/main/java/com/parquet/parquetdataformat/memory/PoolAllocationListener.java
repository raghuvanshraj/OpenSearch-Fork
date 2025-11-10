/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package com.parquet.parquetdataformat.memory;

import org.apache.arrow.memory.AllocationListener;
import org.apache.arrow.memory.AllocationOutcome;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class PoolAllocationListener implements AllocationListener {

    private static final Logger logger = LogManager.getLogger(PoolAllocationListener.class);

    @Override
    public void onPreAllocation(long size) {
        AllocationListener.super.onPreAllocation(size);
    }

    @Override
    public void onChildRemoved(BufferAllocator parentAllocator, BufferAllocator childAllocator) {
        logger.info("Child allocator removed from pool: {} for parent allocator {}", childAllocator.getName(), parentAllocator.getName());
    }

    @Override
    public void onChildAdded(BufferAllocator parentAllocator, BufferAllocator childAllocator) {
        logger.info("Child allocator added to pool: {} for parent allocator {}", childAllocator.getName(), parentAllocator.getName());
    }

    @Override
    public boolean onFailedAllocation(long size, AllocationOutcome outcome) {
        logger.error("Failed to allocate {} bytes, outcome: {}", size, outcome);

        // TODO we might be able to free up some memory here
        return false;
    }

    @Override
    public void onRelease(long size) {
        logger.info("Released {} bytes", size);
    }

    @Override
    public void onAllocation(long size) {
        logger.info("Allocated {} bytes", size);
    }
}
