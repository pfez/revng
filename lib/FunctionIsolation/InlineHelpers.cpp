/// \file InlineHelpers.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionIsolation/InlineHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

// TODO: make sure we do not inline loops

char InlineHelpersPass::ID = 0;

using Register = RegisterPass<InlineHelpersPass>;
static Register X("inline-helpers", "Inline Helpers Pass", true, true);

static bool shouldInline(Function *F) {
  if (F == nullptr)
    return false;

  return F->getSection() == "revng_inline";
}

static CallInst *getCallToInline(Instruction *I) {
  if (auto *Call = dyn_cast<CallInst>(I)) {
    if (shouldInline(Call->getCalledFunction())) {
      return Call;
    }
  }

  return nullptr;
}

static void doInline(CallInst *Call) {
  InlineFunctionInfo IFI;
  auto Result = InlineFunction(Call, IFI, nullptr, false);
  revng_assert(Result);
}

static bool doInline(Function *F) {
  SmallVector<CallInst *, 8> ToInline;

  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (auto *Call = getCallToInline(&I))
        ToInline.push_back(Call);

  for (CallInst *Call : ToInline) {
    doInline(Call);
  }

  return ToInline.size() > 0;
}

class InlineHelpers {
private:
  LLVMContext &C;

public:
  InlineHelpers(Module *M) : C(M->getContext()) {}

  void run(Function *F);

private:
  void wrapCallsToHelpers(Function *F);
};

void InlineHelpers::run(Function *F) {
  // Fixed-point inlining
  while (doInline(F))
    ;
}

bool InlineHelpersPass::runOnFunction(Function &F) {
  if (not FunctionTags::Lifted.isTagOf(&F))
    return false;

  InlineHelpers IH(F.getParent());
  IH.run(&F);
  return true;
}
