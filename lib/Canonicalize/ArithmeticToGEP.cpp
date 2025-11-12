//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <iterator>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/EagerMaterializationRangeIterator.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static constexpr const char *ArithToGEPFlag = "arithmetic-to-gep";

struct ArithmeticToGEPPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  ArithmeticToGEPPass() : FunctionPass(ID) {}

public:
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

static bool isExtractValue(const llvm::Value &V) {
  return isa<llvm::ExtractValueInst>(V)
         or isCallToTagged(&V, FunctionTags::OpaqueExtractValue);
}

static unsigned getExtractValueNumIndices(const llvm::Value &V) {
  revng_assert(isExtractValue(V));

  if (const auto *E = dyn_cast<llvm::ExtractValueInst>(&V))
    return E->getNumIndices();

  // If it's an OpaqueExtractValue, we only ever admit a single index.
  return 1;
}

static const llvm::Value *getAggregateOperand(const llvm::Value &V) {
  revng_assert(isExtractValue(V));
  revng_assert(getExtractValueNumIndices(V) == 1);

  if (const auto *E = dyn_cast<llvm::ExtractValueInst>(&V))
    return E->getAggregateOperand();

  const llvm::CallInst *C = getCallToTagged(&V,
                                            FunctionTags::OpaqueExtractValue);
  return C->getArgOperand(0);
}

static uint64_t getExtractedFieldIndex(const llvm::Value &V) {
  revng_assert(isExtractValue(V));
  revng_assert(getExtractValueNumIndices(V) == 1);

  if (const auto *E = dyn_cast<llvm::ExtractValueInst>(&V))
    return E->getIndices()[0];

  const llvm::CallInst *C = getCallToTagged(&V,
                                            FunctionTags::OpaqueExtractValue);
  const auto *Index = llvm::cast<llvm::ConstantInt>(C->getArgOperand(1));
  return Index->getZExtValue();
}

//
// Helper functions to detect if `llvm::Value`s are pointers, even if their
// `llvm::Type` is not a pointer.
// This is done using additional information coming from Model types and from
// specific instances of `llvm::Value` that can be associated to those types in
// the Model.
// This association is attached to `llvm::Value`s via the revng.is_pointer
// metadata.
//

static bool isArgumentModelPointer(const llvm::Argument *A) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  const llvm::Function *F = A->getParent();
  std::optional<llvm::SmallVector<bool>>
    PointerArguments = getPointerOperandsMetadata(F);
  // The metadata can be missing, if for some reason A is an argument of an
  // llvm::Function that doesn't represent something coming from the binary
  // (e.g. a QEMU helper, or an LLVM intrinsic). But if it's present, the ArgNo
  // must be in range, otherwise something severely wrong is going on.
  revng_assert(not PointerArguments.has_value()
               or A->getArgNo() < PointerArguments.value().size());
  return PointerArguments.has_value()
         and PointerArguments.value()[A->getArgNo()];
}

static bool isNthArgumentModelPointer(const llvm::CallInst *Call,
                                      uint64_t ArgumentIndex) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  if (const llvm::Function *Callee = getCallee(Call))
    return isArgumentModelPointer(Callee->getArg(ArgumentIndex));

  std::optional<llvm::SmallVector<bool>>
    PointerArguments = getPointerOperandsMetadata(Call);
  // The metadata cannot be missing, because this is an indirect call, and
  // indirect calls can only call llvm::Functions that represent something
  // coming from the binary, hence the metadata must be there.
  revng_assert(PointerArguments.has_value());
  revng_assert(ArgumentIndex < PointerArguments.value().size());
  return PointerArguments.value()[ArgumentIndex];
}

static bool returnsModelPointer(const llvm::Function *F) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  std::optional<llvm::SmallVector<bool>>
    PointerReturnValues = getPointerValuesMetadata(F);
  // The metadata can be missing, if for some reason F is an
  // llvm::Function that doesn't represent something coming from the binary
  // (e.g. a QEMU helper, or an LLVM intrinsic).
  return PointerReturnValues.has_value()
         and PointerReturnValues.value().size() == 1
         and PointerReturnValues.value()[0];
}

static bool returnsModelPointer(const llvm::CallInst *Call) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  if (const llvm::Function *Callee = getCallee(Call))
    return returnsModelPointer(Callee);

  std::optional<llvm::SmallVector<bool>>
    PointerReturnValues = getPointerValuesMetadata(Call);
  // The metadata cannot be missing, because this is an indirect call, and
  // indirect calls can only call llvm::Functions that represent something
  // coming from the binary, hence the metadata must be there.
  revng_assert(PointerReturnValues.has_value());
  return PointerReturnValues.value().size() == 1
         and PointerReturnValues.value()[0];
}

static bool isNthReturnValueModelPointer(const llvm::Function *F,
                                         unsigned ReturnValueIndex) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  std::optional<llvm::SmallVector<bool>>
    PointerReturnValues = getPointerValuesMetadata(F);
  revng_assert(PointerReturnValues.has_value());
  revng_assert(ReturnValueIndex < PointerReturnValues.value().size());
  return PointerReturnValues.value()[ReturnValueIndex];
}

static bool isNthReturnValueModelPointer(const llvm::CallInst *Call,
                                         unsigned ReturnValueIndex) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  if (const llvm::Function *Callee = getCallee(Call))
    return isNthReturnValueModelPointer(Callee, ReturnValueIndex);

  std::optional<llvm::SmallVector<bool>>
    PointerReturnValues = getPointerValuesMetadata(Call);
  revng_assert(PointerReturnValues.has_value());
  revng_assert(ReturnValueIndex < PointerReturnValues.value().size());
  return PointerReturnValues.value()[ReturnValueIndex];
}

static bool isExtractedValueModelPointer(const llvm::Value &V) {
  revng_assert(isExtractValue(V));

  const auto *AggregateCall = dyn_cast<llvm::CallInst>(getAggregateOperand(V));
  if (not AggregateCall)
    return false;

  uint64_t FieldIndex = getExtractedFieldIndex(V);
  return isNthReturnValueModelPointer(AggregateCall, FieldIndex);
}

static bool isPointer(const llvm::Value &V) {
  if (V.getType()->isPointerTy())
    return true;

  if (const auto *A = dyn_cast<llvm::Argument>(&V))
    return isArgumentModelPointer(A);

  if (const auto *Call = dyn_cast<llvm::CallInst>(&V))
    if (returnsModelPointer(Call))
      return true;

  if (isExtractValue(V))
    return isExtractedValueModelPointer(V);

  return false;
}

//
// Helper functions to detect if an `llvm::Use` implies that the used
// `llvm::Value` is a pointer, even if its `llvm::Type` is not a pointer.
// This is done using additional information coming from Model types and from
// specific instances of `llvm::User`s` whose uses in certain situations can be
// associated to those types in the Model.
//

static bool isLLVMPointerUse(const llvm::Use &U) {
  llvm::User *TheUser = U.getUser();

  if (const auto *Load = dyn_cast<llvm::LoadInst>(TheUser);
      Load and U.getOperandNo() == Load->getPointerOperandIndex())
    return true;

  if (const auto *Store = dyn_cast<llvm::StoreInst>(TheUser);
      Store and U.getOperandNo() == Store->getPointerOperandIndex())
    return true;

  if (const auto *Call = dyn_cast<llvm::CallInst>(TheUser);
      Call and U->getType()->isPointerTy())
    return true;

  if (const auto *Ret = dyn_cast<llvm::ReturnInst>(TheUser);
      Ret and U->getType()->isPointerTy())
    return true;

  return false;
}

static bool isModelPointerUse(const llvm::Use &U) {
  const llvm::User *TheUser = U.getUser();

  if (not isa<llvm::Instruction>(TheUser))
    return false;

  const llvm::Function &F = *cast<llvm::Instruction>(TheUser)->getFunction();

  if (const auto *Ret = dyn_cast<llvm::ReturnInst>(TheUser))
    return returnsModelPointer(&F);

  const auto *C = dyn_cast<llvm::CallInst>(TheUser);
  if (not C)
    return false;

  // The callee is always a pointer.
  if (C->isCallee(&U))
    return true;

  revng_assert(C->isArgOperand(&U));

  uint64_t ArgNo = C->getArgOperandNo(&U);
  if (isCallToIsolatedFunction(C))
    return isNthArgumentModelPointer(C, ArgNo);

  if (const auto
        *Initializer = getCallToTagged(TheUser,
                                       FunctionTags::StructInitializer)) {
    return isNthReturnValueModelPointer(&F, ArgNo);
  }

  return false;
}

static bool isPointerUse(const llvm::Use &U) {
  return isLLVMPointerUse(U) or isModelPointerUse(U);
}

//
// Helpers for what is considered a local or a global in this file.
//

static bool isLocal(const llvm::Value *V) {
  return isa<llvm::Instruction>(V) or isa<llvm::Argument>(V);
}

static bool isGlobal(const llvm::Value *V) {
  return isa<llvm::ConstantInt>(V) or isa<llvm::Function>(V)
         or isa<llvm::GlobalVariable>(V);
}

//
// Helper concepts for constraining LocalValue constructors, assignments
//

template<class Derived>
concept DerivedFromInstruction = std::derived_from<Derived, llvm::Instruction>;

template<class Derived>
concept DerivedFromArgument = std::derived_from<Derived, llvm::Argument>;

template<class Derived>
concept DerivedFromLocal = DerivedFromInstruction<Derived>
                           or DerivedFromArgument<Derived>;

template<class Derived>
concept ConstDerivedFromLocal = DerivedFromLocal<Derived>
                                and is_const_v<Derived>;

template<class Derived>
concept MutableDerivedFromLocal = DerivedFromInstruction<Derived>
                                  and not is_const_v<Derived>;

// Wrapper for an llvm::Value that for global values only exposes a single use
// at a time. This allows to treat e.g. each use of a constant in a Function as
// if they were effectively separate values.
template<bool IsConst = false>
class LocalValue {

public:
  using ValueType = std::conditional_t<IsConst, const llvm::Value, llvm::Value>;
  using UseType = std::conditional_t<IsConst, const llvm::Use, llvm::Use>;
  using UseIterator = std::conditional_t<IsConst,
                                         llvm::Value::const_use_iterator,
                                         llvm::Value::user_iterator>;

private:
  // The llvm::Value being wrapped.
  ValueType *WrappedValue;
  // When WrappedValue is not DerivedFromLocal, this refers to the single user
  // of WrappedValue that we want to consider.
  UseType *WrappedUse;

public:
  // No need to mark this explicit. Initializing from nullptr explicitly is
  // never ambiguous and can never lead to problems.
  // All other constructors are marked as explicit because they verify
  LocalValue(nullptr_t) : WrappedValue(nullptr), WrappedUse(nullptr) {}
  LocalValue() : LocalValue(nullptr) {}

public:
  LocalValue(const LocalValue &Other) = default;
  LocalValue &operator=(const LocalValue &Other) = default;

  LocalValue(LocalValue &&Other) = default;
  LocalValue &operator=(LocalValue &&Other) = default;

  ~LocalValue() = default;

  // Always enable assigning a Value to a LocalValue.
  template<MutableDerivedFromLocal LocalValueType>
  LocalValue &operator=(LocalValueType *V) {
    return *this = LocalValue<IsConst>(V);
  }

  // Enable assigning a const Value to a LocalValue, only if IsConst.
  template<ConstDerivedFromLocal LocalValueType>
  LocalValue &operator=(LocalValueType *V)
    requires IsConst
  {
    return *this = LocalValue<IsConst>(V);
  }

  // Always enable assigning a Use to a LocalValue.
  LocalValue &operator=(llvm::Use *U) { return *this = LocalValue<IsConst>(U); }

  // Enable assigning a const Use to a LocalValue, only if IsConst.
  LocalValue &operator=(const llvm::Use *U)
    requires IsConst
  {
    return *this = LocalValue<IsConst>(U);
  }

  // Needed by conversion operator
  template<bool Const>
  friend class LocalValue;

  // Enable converting an mutable LocalValue to a const one.
  // This conversion doesn't need to be explicit. In fact it's ergonomic for it
  // not to be, without risks.
  operator LocalValue</* IsConst */ true>()
    requires(not IsConst)
  {
    LocalValue</*IsConst*/ true> Result;
    Result.WrappedValue = this->WrappedValue;
    Result.WrappedUse = this->WrappedUse;
    return Result;
  }

public:
  LocalValue(ValueType *V) : WrappedValue(V), WrappedUse(nullptr) {
    // Ensure this is only called on local values.
    revng_assert(not WrappedValue or isLocal(WrappedValue));
  }

  LocalValue(UseType *U) : WrappedValue(U ? U->get() : nullptr), WrappedUse(U) {
    // Ensure this is only called on global values values.
    revng_assert(not WrappedValue
                 or (isGlobal(WrappedValue) and WrappedUse
                     and isLocal(WrappedUse->getUser())));
  }

public:
  friend std::strong_ordering operator<=>(const LocalValue &LHS,
                                          const LocalValue &RHS) = default;

  friend bool operator==(const LocalValue &LHS,
                         const LocalValue &RHS) = default;

public:
  llvm::SmallVector<UseType *> uses() const {
    llvm::SmallVector<UseType *> Result;
    if (WrappedUse) {
      Result.push_back(WrappedUse);
    } else {
      Result.reserve(WrappedValue->getNumUses());
      for (UseType &U : WrappedValue->uses())
        Result.push_back(&U);
    }
    return Result;
  }

  llvm::SmallVector<LocalValue<IsConst>, 2> users() const {
    llvm::SmallVector<LocalValue<IsConst>, 2> Result;
    llvm::transform(this->uses(), std::back_inserter(Result), [](UseType *U) {
      return LocalValue<IsConst>(U->getUser());
    });
    return Result;
  }

  llvm::SmallVector<LocalValue<IsConst>, 2> operands() const {
    llvm::SmallVector<LocalValue<IsConst>, 2> Result;

    if (WrappedUse)
      return Result;

    llvm::transform(WrappedValue->operands(),
                    std::back_inserter(Result),
                    [this](UseType *Op) {
                      return LocalValue<IsConst>(Op->get());
                    });

    return Result;
  }

public:
  ValueType *value() const { return WrappedValue; }

  llvm::Type *getType() const { return WrappedValue->getType(); }
};

//
// Helper concepts for Value-based deduction guides for LocalValue
//

template<class Derived>
concept DerivedFromValue = std::derived_from<Derived, llvm::Value>;

template<class Derived>
concept ConstDerivedFromValue = std::is_const_v<Derived>
                                and DerivedFromValue<Derived>;

template<class Derived>
concept MutableDerivedFromValue = not std::is_const_v<Derived>
                                  and DerivedFromValue<Derived>;

//
// Value-based deduction guide for LocalValue
//

template<ConstDerivedFromValue ConstValue>
LocalValue(ConstValue *V) -> LocalValue</*IsConst*/ true>;

template<MutableDerivedFromValue MutableValue>
LocalValue(MutableValue *V) -> LocalValue</*IsConst*/ false>;

//
// Helper concepts for Use-based deduction guides for LocalValue
//

template<class Derived>
concept DerivedFromUse = std::derived_from<Derived, llvm::Use>;

template<class Derived>
concept ConstDerivedFromUse = std::is_const_v<Derived>
                              and DerivedFromUse<Derived>;

template<class Derived>
concept MutableDerivedFromUse = not std::is_const_v<Derived>
                                and DerivedFromUse<Derived>;

//
// Use-based deduction guide for LocalValue
//

template<ConstDerivedFromUse ConstUse>
LocalValue(ConstUse *U) -> LocalValue</*IsConst*/ true>;

template<MutableDerivedFromUse MutableUse>
LocalValue(MutableUse *U) -> LocalValue</*IsConst*/ false>;

template<bool IsConst>
struct llvm::GraphTraits<LocalValue<IsConst>> {

private:
  using EagerRange = EagerMaterializationRangeIterator<LocalValue<IsConst>>;

public:
  using NodeRef = LocalValue<IsConst>;
  using ChildIteratorType = EagerRange;

public:
  static ChildIteratorType child_begin(NodeRef N) { return { N.users() }; }

  static ChildIteratorType child_end(NodeRef N) { return { { nullptr } }; }

public:
  static NodeRef getEntryNode(NodeRef V) {
    revng_abort("getEntryNode on LocalValue dataflow graph");
    return LocalValue<IsConst>{ nullptr };
  }
};

template<bool IsConst>
struct llvm::GraphTraits<llvm::Inverse<LocalValue<IsConst>>> {
private:
  using EagerRange = EagerMaterializationRangeIterator<LocalValue<IsConst>>;

public:
  using NodeRef = LocalValue<IsConst>;
  using ChildIteratorType = EagerRange;

public:
  static ChildIteratorType child_begin(NodeRef N) { return { N.operands() }; }

  static ChildIteratorType child_end(NodeRef N) { return { { nullptr } }; }

public:
  static NodeRef getEntryNode(NodeRef V) {
    revng_abort("getEntryNode on Inverse<LocalValue> dataflow graph");
    return LocalValue<IsConst>{ nullptr };
  }
};

static llvm::SmallVector<LocalValue<>, 2> getLocalPointers(llvm::Function &F) {
  llvm::SetVector<LocalValue<>,
                  llvm::SmallVector<LocalValue<>, 2>,
                  llvm::SmallSet<LocalValue<>, 2>>
    LocalPointers;
  for (llvm::Instruction &I : llvm::instructions(F)) {
    for (llvm::Use &U : I.operands()) {
      if (isPointerUse(U)) {
        if (isGlobal(U.get()))
          LocalPointers.insert(LocalValue<>{ &U });
        else if (isLocal(U.get()))
          LocalPointers.insert(LocalValue<>{ U.get() });
      }
    }
  }
  return LocalPointers.takeVector();
}

static llvm::SmallVector<llvm::Use *> snapshotUses(llvm::Instruction *V) {
  llvm::SmallVector<llvm::Use *> Uses;
  const auto ToPointer = [](llvm::Use &U) { return &U; };
  llvm::transform(V->uses(), std::back_inserter(Uses), ToPointer);
  return Uses;
}

static llvm::GetElementPtrInst *makeGEP(revng::NonDebugInfoCheckingIRBuilder &B,
                                        llvm::Value *Base,
                                        llvm::Value *Offset) {

  if (not Base->getType()->isPointerTy()) {
    revng_assert(Base->getType()->isIntegerTy());
    B.CreateIntToPtr(Base, llvm::PointerType::get(B.getContext(), 0));
  }

  auto *GEP = B.CreateGEP(llvm::IntegerType::getInt8Ty(B.getContext()),
                          Base,
                          Offset);
  return cast<llvm::GetElementPtrInst>(GEP);
}

class GEPRewriter {
private:
  revng::NonDebugInfoCheckingIRBuilder B;

public:
  GEPRewriter(llvm::LLVMContext &C) : B{ C } {}

public:
  void replace(LocalValue<> LV) {
    llvm::Value *PointerValue = LV.value();

    if (not LV.getType()->isPointerTy()) {
      setInsertPointAfter(LV);
      PointerValue = B.CreateIntToPtr(PointerValue,
                                      llvm::PointerType::get(B.getContext(),
                                                             0));
    }

    for (llvm::Use *U : LV.uses())
      replaceImpl(U, PointerValue);
  }

private:
  void setInsertPointBeforeUserInstruction(llvm::Use *U) {
    auto *UserInstruction = cast<llvm::Instruction>(U->getUser());
    revng_assert(not isa<llvm::AllocaInst>(UserInstruction));

    if (auto *PHI = dyn_cast<llvm::PHINode>(UserInstruction))
      B.SetInsertPoint(PHI->getIncomingBlock(*U)->getTerminator());
    else
      B.SetInsertPoint(UserInstruction);
  }

  void setInsertPointAfter(llvm::Instruction *I) {
    if (I->isTerminator()) {
      B.SetInsertPoint(I->getParent());
      return;
    }

    auto *NextInstr = &*std::next(I->getIterator());
    if (isa<llvm::PHINode>(NextInstr))
      B.SetInsertPoint(NextInstr->getParent()->getFirstNonPHI());
    else if (isa<llvm::AllocaInst>(NextInstr))
      B.SetInsertPointPastAllocas(NextInstr->getParent()->getParent());
    else
      B.SetInsertPoint(NextInstr);
  }

  void setInsertPointAfter(LocalValue<> LV) {
    llvm::Value *V = LV.value();
    if (isGlobal(V)) {
      B.SetInsertPoint(cast<llvm::Instruction>(LV.uses().front()->getUser()));
    } else if (auto *A = dyn_cast<llvm::Argument>(V)) {
      B.SetInsertPointPastAllocas(A->getParent());
    } else {
      setInsertPointAfter(cast<llvm::Instruction>(V));
    }
  }

  RecursiveCoroutine<void> replaceImpl(llvm::Use *U, llvm::Value *BasePointer) {
    revng_assert(BasePointer->getType()->isIntOrPtrTy());

    auto *UserInstruction = cast<llvm::Instruction>(U->getUser());

    switch (auto Opcode = UserInstruction->getOpcode(); Opcode) {

    case llvm::Instruction::Add: {

      // TODO bail out in case of add with negative constant

      setInsertPointBeforeUserInstruction(U);

      auto *GEP = makeGEP(B,
                          BasePointer,
                          UserInstruction->getOperand(U->getOperandNo() == 0 ?
                                                        1 :
                                                        0));
      for (llvm::Use *GEPUse : snapshotUses(UserInstruction))
        rc_recur replaceImpl(GEPUse, GEP);

    } break;

    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::BitCast: {

      for (llvm::Use *CastUse : snapshotUses(UserInstruction))
        rc_recur replaceImpl(CastUse, BasePointer);

    } break;

    case llvm::Instruction::GetElementPtr: {
      // Don't create any GEP in this case, since one is already there.
      // Just recur on all of the GEP's uses if this is the address operand. Or
      // fall back on the default if this is one of the indices.
      auto PointerOpIndex = llvm::GetElementPtrInst::getPointerOperandIndex();
      if (U->getOperandNo() == PointerOpIndex) {

        for (llvm::Use *GEPUse : snapshotUses(UserInstruction))
          rc_recur replaceImpl(GEPUse, UserInstruction);

        break;
      }
    }
      [[fallthrough]];

    default: {
      // We've reached the end of the linear path that can be rewritten as a
      // GEP. Just cast the the value to the proper type if necessary.
      llvm::Type *UseType = U->get()->getType();
      revng_assert(UseType->isIntOrPtrTy());
      if (not UseType->isPointerTy()) {
        setInsertPointBeforeUserInstruction(U);
        BasePointer = B.CreatePtrToInt(BasePointer, UseType);
      }
      U->set(BasePointer);
    }
    }
    rc_return;
  }
};

bool ArithmeticToGEPPass::runOnFunction(llvm::Function &F) {

  llvm::SmallVector<LocalValue<>> ObviousPointers = getLocalPointers(F);

  GEPRewriter Rewriter(F.getContext());
  for (const LocalValue<> &PointerValue : ObviousPointers) {
    Rewriter.replace(PointerValue);
  }

  return not ObviousPointers.empty();

  // we should also compute the likely pointers based on the transitive pointer
  // uses.

  // remove the following test code

  isPointer(F);

  LocalValue<true> v1{ declval<llvm::Instruction *>() };
  LocalValue<true> v2{ declval<const llvm::Instruction *>() };
  LocalValue<false> v3{ declval<llvm::Argument *>() };
  // The following does not compile because const llvm::Argument cannot be use
  // to initialize a mutable LocalValue
  // LocalValue<false> v4{declval<const llvm::Argument*>()};

  // These work because we have the assignment operator from DerivedFromLocal *
  v1 = declval<llvm::Instruction *>();
  v1 = declval<const llvm::Instruction *>();
  // The following doesn't work because we don't have assignment operator for
  // DerivedFromValue * that are not also DerivedFromLocal *.
  // v3 = declval<llvm::ConstantInt *>();
  // The following does not compile because const llvm::Instruction cannot be
  // use to initialize a mutable LocalValue
  // v3 = declval<const llvm::Instruction *>();

  LocalValue w1{ declval<llvm::Instruction *>() };
  LocalValue w2{ declval<const llvm::Instruction *>() };

  // This works because we have the assignment operator
  w2 = declval<llvm::Instruction *>();
  w2 = declval<const llvm::Instruction *>();
  w1 = declval<llvm::Instruction *>();
  // The following doesn't compile because we can't assign a const
  // llvm::Instruction to a mutable LocalValue
  // w1 = declval<const llvm::Instruction *>();

  // This does not compile because it assigns a ConstLocalValue to a
  // mutable LocalValue
  // w1 = v1;
  // The opposite works fine though.
  v1 = w1;

  LocalValue<true> gv1{ declval<llvm::Use *>() };
  LocalValue<true> gv2{ declval<const llvm::Use *>() };
  LocalValue<false> gv3{ declval<llvm::Use *>() };
  // The following does not compile because const llvm::Use cannot be use
  // to initialize a mutable LocalValue
  // LocalValue<false> v4{declval<const llvm::Use*>()};
  LocalValue gw1{ declval<llvm::Use *>() };
  LocalValue gw2{ declval<const llvm::Use *>() };

  // This works because we have the assignment operator
  gw2 = declval<llvm::Use *>();
  gw2 = declval<const llvm::Use *>();
  gw1 = declval<llvm::Use *>();
  // The following doesn't compile because we can't assign a const
  // llvm::Instruction to a mutable LocalValue
  // gw1 = declval<const llvm::Use *>();

  // This does not compile because it assigns a ConstLocalValue to a
  // mutable LocalValue
  // gw1 = gv1;
  // The opposite works fine though.
  gv1 = w1;

  return false;
}

char ArithmeticToGEPPass::ID = 0;

static constexpr const char *Description = "Arithmetic-to-i8-GEP replacement";

static llvm::RegisterPass<ArithmeticToGEPPass> X{ ArithToGEPFlag,
                                                  Description,
                                                  false,
                                                  false };
