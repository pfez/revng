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
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/EagerMaterializationRangeIterator.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static constexpr const char *ArithmeticToGEPFlag = "arithmetic-to-gep";

static Logger FindLog{ "arithmetic-to-gep-find-pointers" };

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
  // The metadata can be missing, if F doesn't represent a model::Function nor a
  // model::DynamicFunction. Otherwise it must be well formed.
  revng_assert(not PointerReturnValues.has_value()
               or ReturnValueIndex < PointerReturnValues.value().size());
  return PointerReturnValues.has_value()
         and PointerReturnValues.value()[ReturnValueIndex];
}

static bool isNthReturnValueModelPointer(const llvm::CallInst *Call,
                                         unsigned ReturnValueIndex) {
  // FIXME TODO: this should handle SPTAR too. make sure it does.
  if (const llvm::Function *Callee = getCallee(Call))
    return isNthReturnValueModelPointer(Callee, ReturnValueIndex);

  std::optional<llvm::SmallVector<bool>>
    PointerReturnValues = getPointerValuesMetadata(Call);
  // The metadata cannot be missing, because this is an indirect call, and
  // indirect calls can only call llvm::Functions that represent something
  // coming from the binary, hence the metadata must be there.
  revng_assert(PointerReturnValues.has_value());
  revng_assert(ReturnValueIndex < PointerReturnValues.value().size());
  return PointerReturnValues.has_value()
         and PointerReturnValues.value()[ReturnValueIndex];
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

static bool isPointer(const llvm::Use &U) {
  return isPointer(*U.get());
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

[[maybe_unused]] static bool isLocal(const llvm::Use *U) {
  return isLocal(U->get());
}

static bool isGlobal(const llvm::Value *V) {
  return isa<llvm::Function>(V) or isa<llvm::GlobalVariable>(V)
         or isa<llvm::Constant>(V);
}

static bool isGlobal(const llvm::Use *U) {
  return isGlobal(U->get());
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
  using UserType = std::conditional_t<IsConst, const llvm::User, llvm::User>;
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
  bool verify() const {
    if (WrappedUse) {
      // If the Use is not null, so must be the Value, and it must be a global,
      // and the Use must be using V.
      return WrappedValue and isGlobal(WrappedValue)
             and WrappedUse->get() == WrappedValue;
    }

    if (WrappedValue) {
      // If the Value is not null, and the Use is missing, the Value must be a
      // local.
      return isLocal(WrappedValue);
    }

    return true;
  }

private:
  LocalValue(ValueType *V, UseType *U) : WrappedValue(V), WrappedUse(U) {
    revng_assert(this->verify());
  }

public:
  // No need to mark this explicit. Initializing from nullptr explicitly is
  // never ambiguous and can never lead to problems.
  // All other constructors are marked as explicit because they verify
  LocalValue(nullptr_t) : LocalValue(nullptr, nullptr) {}
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
  LocalValue(ValueType *V) : LocalValue(V, nullptr) {}
  LocalValue(UseType *U) :
    LocalValue(U ? U->get() : nullptr, (U and isGlobal(U)) ? U : nullptr) {}

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

  llvm::SmallVector<LocalValue<IsConst>, 2> maybePointerOperands() const {
    llvm::SmallVector<LocalValue<IsConst>, 2> Result;

    if (WrappedUse)
      return Result;

    if (not isLocal(WrappedValue))
      return Result;

    if (auto *WrappedUserValue = dyn_cast<UserType>(WrappedValue)) {

      static constexpr auto UseToLocalValue = [](UseType &Operand) {
        return LocalValue<IsConst>(&Operand);
      };

      llvm::transform(llvm::make_filter_range(WrappedUserValue->operands(),
                                              hasDataflowToPointerOperand),
                      std::back_inserter(Result),
                      UseToLocalValue);
    }

    return Result;
  }

public:
  ValueType *value() const { return WrappedValue; }

  UseType *use() const { return WrappedUse; }

  llvm::Type *getType() const { return WrappedValue->getType(); }

private:
  static bool hasDataflowToPointerOperand(UseType &OperandUse) {

    ValueType *Operand = OperandUse.get();
    UserType *User = OperandUse.getUser();

    if (isa<llvm::Constant>(User))
      return false;

    if (auto *I = dyn_cast<llvm::Instruction>(User)) {
      revng_assert(not I->isTerminator());
      switch (I->getOpcode()) {

      case llvm::Instruction::PHI:
        return not isa<llvm::BlockAddress>(Operand);

      case llvm::Instruction::Freeze:
      case llvm::Instruction::IntToPtr:
      case llvm::Instruction::PtrToInt:
      case llvm::Instruction::BitCast:
        return true;

      case llvm::Instruction::Add:
        return true;

      case llvm::Instruction::Sub:
        return OperandUse.getOperandNo() == 0;

      case llvm::Instruction::GetElementPtr:
        return OperandUse.getOperandNo() == 0;

      case llvm::Instruction::AShr:
      case llvm::Instruction::LShr:
      case llvm::Instruction::Shl:
        return OperandUse.getOperandNo() == 0;

      case llvm::Instruction::Select:
        return OperandUse.getOperandNo() != 0;

      case llvm::Instruction::Alloca:
      case llvm::Instruction::Load:
      case llvm::Instruction::Store:
      case llvm::Instruction::Mul:
      case llvm::Instruction::SDiv:
      case llvm::Instruction::UDiv:
      case llvm::Instruction::SRem:
      case llvm::Instruction::URem:
      case llvm::Instruction::ICmp:
      default:
        return false;
      }
    }

    return false;
  }
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

//
// Graph traits for LocalValue
//

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
  static ChildIteratorType child_begin(NodeRef N) {
    return { N.maybePointerOperands() };
  }

  static ChildIteratorType child_end(NodeRef N) { return { { nullptr } }; }

public:
  static NodeRef getEntryNode(NodeRef V) {
    return LocalValue<IsConst>{ nullptr };
  }
};

class PointersFinder {
  llvm::ModuleSlotTracker MST;

public:
  PointersFinder(llvm::Function &F) : MST(F.getParent()) {}

  llvm::SmallVector<LocalValue<>> findPointers(llvm::Function &F) {
    llvm::SmallVector<LocalValue<>> Result;
    Result = findObviousPointers(F);
    Result.append(findLikelyPointers(F));
    return Result;
  }

private:
  llvm::SmallVector<LocalValue<>, 2> findObviousPointers(llvm::Function &F) {
    revng_log(FindLog, "findPointers: " << F.getName());
    LoggerIndent Indent{ FindLog };

    llvm::SetVector<LocalValue<>,
                    llvm::SmallVector<LocalValue<>, 2>,
                    llvm::SmallSet<LocalValue<>, 2>>
      LocalPointers;

    for (llvm::Argument &A : F.args()) {
      if (isPointer(A)) {
        revng_log(FindLog, "Pointer Argument: " << dumpToString(A, MST));
        LocalPointers.insert(&A);
      }
    }

    for (llvm::Instruction &I : llvm::instructions(F)) {
      if (isPointer(I)) {
        revng_log(FindLog, "Pointer Instruction: " << dumpToString(I, MST));
        LocalPointers.insert(&I);
      }
    }

    return LocalPointers.takeVector();
  }

  llvm::SmallVector<LocalValue<>, 2> findLikelyPointers(llvm::Function &F) {
    revng_log(FindLog, "findLikelyPointers: " << F.getName());
    LoggerIndent Indent{ FindLog };

    llvm::DenseSet<llvm::Use *> PointerUses;
    {
      revng_log(FindLog, "Initial pointer uses");
      LoggerIndent Indent{ FindLog };
      for (llvm::Instruction &I : llvm::instructions(F)) {
        for (llvm::Use &U : I.operands()) {
          if (isPointer(U) or isPointerUse(U)) {
            revng_log(FindLog, "Instruction: " << dumpToString(I, MST));
            LoggerIndent IndentI{ FindLog };
            revng_log(FindLog,
                      "has pointer operand: " << dumpToString(*U, MST));
            PointerUses.insert(&U);
          }
        }
      }
    }

    llvm::SetVector<LocalValue<>,
                    llvm::SmallVector<LocalValue<>, 2>,
                    llvm::SmallSet<LocalValue<>, 2>>
      UsedAsPointers;

    revng_log(FindLog, "Propagate pointer uses backwards");
    {
      LoggerIndent IndentPropagation{ FindLog };

      for (llvm::Use *U : PointerUses) {
        llvm::SmallVector<LocalValue<>, 8> Worklist;
        if (LocalValue<> LV{ U }; UsedAsPointers.insert(LV))
          Worklist.push_back(LV);

        while (not Worklist.empty()) {
          LocalValue<> LV = Worklist.pop_back_val();

          revng_log(FindLog,
                    "Considering value:" << dumpToString(*LV.value(), MST));
          if (LV.use())
            revng_log(FindLog,
                      "used in:" << dumpToString(*LV.use()->getUser(), MST));

          LoggerIndent IndentOperands{ FindLog };

          for (LocalValue<> O : LV.maybePointerOperands()) {
            if (UsedAsPointers.insert(O)) {
              revng_log(FindLog,
                        "found new pointer operand: "
                          << dumpToString(*O.value(), MST));
              Worklist.push_back(O);
            }
          }
        }
      }
    }

    if (FindLog.isEnabled()) {
      for (LocalValue<> LV : UsedAsPointers) {
        revng_log(FindLog,
                  "value used as pointer: " << dumpToString(*LV.value(), MST));
      }
    }

    return UsedAsPointers.takeVector();
  }
};

static bool foldPointerCasts(llvm::Function &F) {
  using WTVH = llvm::WeakTrackingVH;
  llvm::SmallVector<WTVH, 8> Dead;

  for (llvm::Instruction &I : llvm::instructions(F)) {

    if (auto *PtrToInt = dyn_cast<llvm::PtrToIntInst>(&I)) {
      for (auto &U : PtrToInt->uses()) {

        if (auto *IntToPtr = dyn_cast<llvm::IntToPtrInst>(U.getUser())) {
          llvm::Value *Pointer = PtrToInt->getOperand(0);
          if (Pointer->getType() != IntToPtr->getType())
            continue;

          IntToPtr->replaceAllUsesWith(Pointer);
          Dead.push_back(IntToPtr);
        }
      }

    } else if (auto *IntToPtr = dyn_cast<llvm::IntToPtrInst>(&I)) {
      for (auto &U : IntToPtr->uses()) {

        if (auto *PtrToInt = dyn_cast<llvm::PtrToIntInst>(U.getUser())) {
          llvm::Value *Integer = IntToPtr->getOperand(0);
          if (Integer->getType() != PtrToInt->getType())
            continue;

          PtrToInt->replaceAllUsesWith(Integer);
          Dead.push_back(PtrToInt);
        }
      }
    }
  }

  RecursivelyDeleteTriviallyDeadInstructionsPermissive(Dead);

  return not Dead.empty();
}

class GEPRewriter {
private:
  llvm::Function &F;
  revng::NonDebugInfoCheckingIRBuilder B;

  // Collect all the llvm::Values that had at least one of their uses changed
  // across all calls to replace().
  // This is useful to call DCE in the destructor, given that any of those can
  // now be dead code.
  using WTVH = llvm::WeakTrackingVH;
  llvm::SetVector<WTVH, llvm::SmallVector<WTVH, 8>, llvm::SmallSet<WTVH, 8>>
    HadUsesReplaced;

public:
  GEPRewriter(llvm::Function &F) : F{ F }, B{ F.getContext() } {}

  ~GEPRewriter() {
    llvm::SmallVector<llvm::WeakTrackingVH> R = HadUsesReplaced.takeVector();
    RecursivelyDeleteTriviallyDeadInstructionsPermissive(R);
    foldPointerCasts(F);
  }

public:
  bool replace(LocalValue<> LV) {
    bool Changed = false;

    for (llvm::Use *U : LV.uses())
      Changed |= rc_eval(replaceImpl(U, LV.value()));

    return Changed;
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

  llvm::Instruction *replaceAddWithGEP(llvm::Use *PointerOperandInAdd,
                                       llvm::Value *BasePointer) {

    auto *Add = cast<llvm::Instruction>(PointerOperandInAdd->getUser());
    revng_assert(Add->getOpcode() == llvm::Instruction::Add);
    B.SetInsertPoint(Add);

    unsigned PointerOpIndex = PointerOperandInAdd->getOperandNo();
    unsigned OffsetOpIndex = PointerOpIndex == 0 ? 1 : 0;
    auto *Offset = Add->getOperand(OffsetOpIndex);

    auto *Pointer = PointerOperandInAdd->get();
    if (Pointer != BasePointer) {
      // We have traversed a bunch of casts, so that Pointer is obtained from
      // BasePointer via just casts.
      // What we want to do here is to make use BasePointer as pointer operand
      // of the GEP we're going to create.
      // It might be necessary to add a IntToPtr cast first.
      Pointer = BasePointer;
    }

    auto *PointerType = llvm::PointerType::get(B.getContext(), 0);
    auto *IntToPtr = B.CreateIntToPtr(Pointer, PointerType);

    auto *Int8 = llvm::IntegerType::getInt8Ty(B.getContext());
    auto *GEP = B.CreateGEP(Int8, IntToPtr, Offset);

    // FIXME TODO potrei fare di meglio qui, tipo creare la PtrToInt solo quando
    // necessario, cioè solo sugli usi che non sono inttoptr. ma in pratica
    // anche se non lo faccio
    auto *AddType = Add->getType();
    auto *GEPToInt = cast<llvm::Instruction>(B.CreatePtrToInt(GEP, AddType));

    Add->replaceAllUsesWith(GEPToInt);
    HadUsesReplaced.insert(Add);

    return GEPToInt;
  }

  llvm::SmallVector<llvm::Use *> snapshotUses(llvm::Instruction *V) const {
    llvm::SmallVector<llvm::Use *> Uses;
    const auto ToPointer = [](llvm::Use &U) { return &U; };
    llvm::transform(V->uses(), std::back_inserter(Uses), ToPointer);
    return Uses;
  }

  RecursiveCoroutine<bool> replaceImpl(llvm::Use *U, llvm::Value *BasePointer) {
    revng_assert(BasePointer->getType()->isIntOrPtrTy());
    revng_assert(U->get()->getType()->isIntOrPtrTy());

    bool Changed = false;

    auto *UserInstruction = cast<llvm::Instruction>(U->getUser());

    switch (auto Opcode = UserInstruction->getOpcode(); Opcode) {

    case llvm::Instruction::Add: {

      // FIXME TODO bail out in case of add with negative constant

      auto *GEPCastedToInt = replaceAddWithGEP(U, BasePointer);
      Changed = true;

      for (llvm::Use *IntUse : snapshotUses(GEPCastedToInt))
        Changed |= rc_recur replaceImpl(IntUse, GEPCastedToInt);

    } break;

    case llvm::Instruction::Freeze:
    case llvm::Instruction::IntToPtr:
    case llvm::Instruction::PtrToInt:
    case llvm::Instruction::BitCast: {

      for (llvm::Use *CastUse : snapshotUses(UserInstruction))
        Changed |= rc_recur replaceImpl(CastUse, BasePointer);

    } break;

    case llvm::Instruction::GetElementPtr: {
      // Don't create any GEP in this case, since one is already there.
      // Just recur on all of the GEP's uses if this is the address operand. Or
      // fall back on the default if this is one of the indices.
      auto *GEP = cast<llvm::GetElementPtrInst>(UserInstruction);
      unsigned PointerOpNo = llvm::GetElementPtrInst::getPointerOperandIndex();
      if (U->getOperandNo() == PointerOpNo) {
        for (llvm::Use *GEPUse : snapshotUses(GEP))
          Changed |= rc_recur replaceImpl(GEPUse, GEP);

        break; // break from switch. all operands do fall through to default.
      }
    }
      [[fallthrough]];
    default: {
      // We've reached the end of the linear path that can be rewritten as a
      // GEP. Just cast back the value to the proper integer or pointer type if
      // necessary.
      if (U->get() != BasePointer) {
        auto *UseType = U->get()->getType();
        auto *BasePointerType = BasePointer->getType();
        if (BasePointerType != UseType) {
          setInsertPointBeforeUserInstruction(U);
          if (BasePointerType->isPointerTy()) {
            BasePointer = B.CreatePtrToInt(BasePointer, UseType);
          } else {
            BasePointer = B.CreateIntToPtr(BasePointer, UseType);
          }
        }
        auto *OldOperand = U->get();
        U->set(BasePointer);
        HadUsesReplaced.insert(OldOperand);
        Changed = true;
      }
    }
    }
    rc_return Changed;
  }
};

static void crashOnPHINode(const llvm::Function &F) {
  for (const llvm::Instruction &I : llvm::instructions(F)) {
    if (isa<llvm::PHINode>(I)) {
      std::string Message = "Unexpected PHINode in Function: ";
      Message += F.getName().str();
      revng_abort(Message.c_str());
    }
  }
}

bool ArithmeticToGEPPass::runOnFunction(llvm::Function &F) {

  crashOnPHINode(F);

  PointersFinder Finder(F);
  llvm::SmallVector<LocalValue<>> Pointers = Finder.findPointers(F);

  if (Pointers.empty())
    return false;

  GEPRewriter Rewriter(F);

  std::set<const LocalValue<>> Replaced;
  for (const LocalValue<> &PointerValue : Pointers) {

    // This can happen because a likely pointer may also be an obvious
    // pointer.
    bool New = Replaced.insert(PointerValue).second;
    if (not New)
      continue;

    Rewriter.replace(PointerValue);
  }

  return true;

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

static llvm::RegisterPass<ArithmeticToGEPPass> X{ ArithmeticToGEPFlag,
                                                  Description,
                                                  false,
                                                  false };
