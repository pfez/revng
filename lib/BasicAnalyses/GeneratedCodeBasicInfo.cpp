/// \file GeneratedCodeBasicInfo.cpp
/// \brief Implements the GeneratedCodeBasicInfo pass which provides basic
///        information about the translated code (e.g., which CSV is the PC).

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>
#include <set>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/Debug.h"

using namespace llvm;

char GeneratedCodeBasicInfo::ID = 0;
using RegisterGCBI = RegisterPass<GeneratedCodeBasicInfo>;
static RegisterGCBI X("gcbi", "Generated Code Basic Info", true, true);

bool GeneratedCodeBasicInfo::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");
  NewPC = M.getFunction("newpc");
  if (NewPC != nullptr) {
    MetaAddressStruct = cast<StructType>(NewPC->arg_begin()->getType());
  }

  revng_log(PassesLog, "Starting GeneratedCodeBasicInfo");

  RootFunction = &F;

  const char *MDName = "revng.input.architecture";
  NamedMDNode *InputArchMD = M.getOrInsertNamedMetadata(MDName);
  auto *Tuple = dyn_cast<MDTuple>(InputArchMD->getOperand(0));

  QuickMetadata QMD(M.getContext());

  {
    unsigned Index = 0;
    StringRef ArchTypeName = QMD.extract<StringRef>(Tuple, Index++);
    ArchType = Triple::getArchTypeForLLVMName(ArchTypeName);
    InstructionAlignment = QMD.extract<uint32_t>(Tuple, Index++);
    DelaySlotSize = QMD.extract<uint32_t>(Tuple, Index++);
    PC = M.getGlobalVariable(QMD.extract<StringRef>(Tuple, Index++), true);
    SP = M.getGlobalVariable(QMD.extract<StringRef>(Tuple, Index++), true);
    auto Operands = QMD.extract<MDTuple *>(Tuple, Index++)->operands();
    for (const MDOperand &Operand : Operands) {
      StringRef Name = QMD.extract<StringRef>(Operand.get());
      revng_assert(Name != "pc", "PC should not be considered an ABI register");
      GlobalVariable *CSV = M.getGlobalVariable(Name, true);
      ABIRegisters.push_back(CSV);
      ABIRegistersSet.insert(CSV);
    }
  }

  Type *PCType = PC->getType()->getPointerElementType();
  PCRegSize = M.getDataLayout().getTypeAllocSize(PCType);

  for (BasicBlock &BB : F) {
    if (!BB.empty()) {
      switch (getType(&BB)) {
      case BlockType::RootDispatcherBlock:
        revng_assert(Dispatcher == nullptr);
        Dispatcher = &BB;
        break;

      case BlockType::DispatcherFailureBlock:
        revng_assert(DispatcherFail == nullptr);
        DispatcherFail = &BB;
        break;

      case BlockType::AnyPCBlock:
        revng_assert(AnyPC == nullptr);
        AnyPC = &BB;
        break;

      case BlockType::UnexpectedPCBlock:
        revng_assert(UnexpectedPC == nullptr);
        UnexpectedPC = &BB;
        break;

      case BlockType::JumpTargetBlock: {
        auto *Call = cast<CallInst>(&*BB.begin());
        revng_assert(Call->getCalledFunction()->getName() == "newpc");
        JumpTargets[MetaAddress::fromConstant(Call->getArgOperand(0))] = &BB;
        break;
      }
      case BlockType::RootDispatcherHelperBlock:
      case BlockType::IndirectBranchDispatcherHelperBlock:
      case BlockType::EntryPoint:
      case BlockType::ExternalJumpsHandlerBlock:
      case BlockType::TranslatedBlock:
        // Nothing to do here
        break;
      }
    }
  }

  if (auto *NamedMD = M.getNamedMetadata("revng.csv")) {
    auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
    for (const MDOperand &Operand : Tuple->operands()) {
      auto *CSV = cast<GlobalVariable>(QMD.extract<Constant *>(Operand.get()));
      CSVs.push_back(CSV);
    }
  }

  revng_log(PassesLog, "Ending GeneratedCodeBasicInfo");

  return false;
}

GeneratedCodeBasicInfo::Successors
GeneratedCodeBasicInfo::getSuccessors(BasicBlock *BB) const {
  Successors Result;

  df_iterator_default_set<BasicBlock *> Visited;
  Visited.insert(AnyPC);
  Visited.insert(UnexpectedPC);
  for (BasicBlock *Block : depth_first_ext(BB, Visited)) {
    for (BasicBlock *Successor : successors(Block)) {
      MetaAddress Address = getBasicBlockPC(Successor);
      const auto IBDHB = BlockType::IndirectBranchDispatcherHelperBlock;
      if (Address.isValid()) {
        Visited.insert(Successor);
        Result.Addresses.insert(Address);
      } else if (Successor == AnyPC) {
        Result.AnyPC = true;
      } else if (Successor == UnexpectedPC) {
        Result.UnexpectedPC = true;
      } else if (getType(Successor) == IBDHB) {
        // Ignore
      } else {
        Result.Other = true;
      }
    }
  }

  return Result;
}
