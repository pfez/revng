#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Run LLVM's `strip-dead-debug-info`, then put `llvm.dbg.cu` back in order
///
/// When that pass drops a dead compile unit it rewrites the whole
/// `llvm.dbg.cu` list out of a container of pointers, so as soon as more than
/// one compile unit survives the order it writes depends on where those
/// metadata nodes happen to sit in memory. Every later pipe then sees a
/// different module, and the types the data layout analysis recovers change
/// from one run to the next.
class StableStripDeadDebugInfo : public llvm::ModulePass {
public:
  static char ID;

public:
  StableStripDeadDebugInfo() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
