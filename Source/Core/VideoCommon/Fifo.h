// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <vector>
#include <queue>
#include <mutex>
#include <unordered_map>
#include "Common/Assert.h"
#include "Common/CommonTypes.h"
#include "Common/SPSCQueue.h"
#include "Core/HW/GPFifo.h"
#include "VideoCommon/DataReader.h"
#include "Statistics.h"
#include "Common/Align.h"

class PointerWrap;

namespace Fifo
{
void Init();
void Shutdown();
void Prepare();  // Must be called from the CPU thread.
void DoState(PointerWrap& f);

// Used for diagnostics.
enum class SyncGPUReason
{
  Other,
  Wraparound,
  EFBPoke,
  PerfQuery,
  BBox,
  Swap,
  AuxSpace,
};

// In single core mode, this runs the GPU for a single slice.
// In dual core mode, this synchronizes with the GPU thread.
void SyncGPUForRegisterAccess();

void RunGpu();
bool AtBreakpoint();
void ResetVideoBuffer();

}  // namespace Fifo
