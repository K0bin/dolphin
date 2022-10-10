// Copyright 2014 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <queue>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "Common/CommonTypes.h"
#include "Common/Flag.h"
#include "Common/Align.h"
#include "Common/Assert.h"
#include "Core/HW/GPFifo.h"
#include "VideoCommon/DataReader.h"
#include "VideoCommon/Statistics.h"

namespace GPUThread {

    void Init();

    void Run();

    void Exit();

    bool IsActive();

    void Wake();

    void FlushFifoChunk();

    void FlushFifoChunkIfNecessary();

    void ProcessGPUChunk();

    struct FifoEntry {
        u32 start;
        u32 length;
    };

    // One FifoChunk is equivalent to one invocation of RunGPUonCPU
    class FifoChunk {
    public:
        FifoChunk() = default;

        FifoChunk(const FifoChunk &other) = delete;

        FifoChunk(FifoChunk &&other) noexcept;

        ~FifoChunk();

        FifoChunk &operator=(FifoChunk &&other) noexcept;

        void Reset();

        void PushFifoData(const u8 *src, u32 length);

        void CopyAuxData(u32 guest_address, const u8 *src, u32 length);

        u8 *AuxData(u32 guest_address);

        bool IsEmpty() const {
          return fifo_entries.empty();
        }

        DataReader NextFifoReader() {
          if (fifo_index >= fifo_entries.size())
            return DataReader(nullptr, nullptr);

          const FifoEntry &entry = fifo_entries[fifo_index++];
          return DataReader(data + entry.start, data + entry.start + entry.length);
        }

    private:
        u8 *data = nullptr;
        u32 data_capacity = 0;
        u8 *aux_data = nullptr;
        u32 aux_data_capacity = 0;
        u32 aux_data_length = 0;
        std::unordered_map<u32, u32> memory_offsets;
        std::vector<FifoEntry> fifo_entries;
        u32 fifo_index = 0;
    };

    class FifoThreadContext {
    public:
        void Flush();

        bool PopReadChunk();

        FifoChunk &WriteChunk() {
          return m_write_chunk;
        }

        FifoChunk &ReadChunk() {
          return m_read_chunk;
        }

    private:
        FifoChunk m_write_chunk;
        FifoChunk m_read_chunk;

        std::queue<FifoChunk> m_submit_queue;
        std::queue<FifoChunk> m_submit_queue_b;
        std::vector<FifoChunk> m_free_list;
        std::vector<FifoChunk> m_free_list_b;

        std::mutex m_submit_mutex;
        std::mutex m_submit_mutex_b;
        std::mutex m_free_list_mutex;
        std::mutex m_free_list_mutex_b;

    };

    FifoChunk& FifoWriteChunk();

    FifoChunk& FifoReadChunk();
}

