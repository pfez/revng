//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <random>
#include <type_traits>

#include "llvm/ADT/SmallSet.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"

using llvm::cast;
using llvm::dyn_cast_or_null;

namespace model {

constexpr std::array<const char *, 93> CReservedKeywords = {
  // C reserved keywords
  "auto",
  "break",
  "case",
  "char",
  "const",
  "continue",
  "default",
  "do",
  "double",
  "else",
  "enum",
  "extern",
  "float",
  "for",
  "goto",
  "if",
  "inline", // Since C99
  "int",
  "long",
  "register",
  "restrict", // Since C99
  "return",
  "short",
  "signed",
  "sizeof",
  "static",
  "struct",
  "switch",
  "typedef",
  "union",
  "unsigned",
  "volatile",
  "while",
  "_Alignas", // Since C11
  "_Alignof", // Since C11
  "_Atomic", // Since C11
  "_Bool", // Since C99
  "_Complex", // Since C99
  "_Decimal128", // Since C23
  "_Decimal32", // Since C23
  "_Decimal64", // Since C23
  "_Generic", // Since C11
  "_Imaginary", // Since C99
  "_Noreturn", // Since C11
  "_Static_assert", // Since C11
  "_Thread_local", // Since C11
  // Convenience macros
  "alignas",
  "alignof",
  "bool",
  "complex",
  "imaginary",
  "noreturn",
  "static_assert",
  "thread_local",
  // Convenience macros for atomic types
  "atomic_bool",
  "atomic_char",
  "atomic_schar",
  "atomic_uchar",
  "atomic_short",
  "atomic_ushort",
  "atomic_int",
  "atomic_uint",
  "atomic_long",
  "atomic_ulong",
  "atomic_llong",
  "atomic_ullong",
  "atomic_char16_t",
  "atomic_char32_t",
  "atomic_wchar_t",
  "atomic_int_least8_t",
  "atomic_uint_least8_t",
  "atomic_int_least16_t",
  "atomic_uint_least16_t",
  "atomic_int_least32_t",
  "atomic_uint_least32_t",
  "atomic_int_least64_t",
  "atomic_uint_least64_t",
  "atomic_int_fast8_t",
  "atomic_uint_fast8_t",
  "atomic_int_fast16_t",
  "atomic_uint_fast16_t",
  "atomic_int_fast32_t",
  "atomic_uint_fast32_t",
  "atomic_int_fast64_t",
  "atomic_uint_fast64_t",
  "atomic_intptr_t",
  "atomic_uintptr_t",
  "atomic_size_t",
  "atomic_ptrdiff_t",
  "atomic_intmax_t",
  "atomic_uintmax_t",
  // C Extensions
  "_Pragma",
  "asm",
};

constexpr std::array<const char *, 21> PrimitiveTypeNames = {
  "uint8_t",   "uint16_t",   "uint32_t",  "uint64_t",  "uint128_t", "int8_t",
  "int16_t",   "int32_t",    "int64_t",   "int128_t",  "bool8_t",   "bool16_t",
  "bool32_t",  "bool64_t",   "bool128_t", "float16_t", "float32_t", "float64_t",
  "float80_t", "float128_t", "void",
};

static inline constexpr std::array<const char *, 10> IntegralTypeNames = {
  "uint8_t", "uint16_t", "uint32_t", "uint64_t", "uint128_t",
  "int8_t",  "int16_t",  "int32_t",  "int64_t",  "int128_t",
};

static inline bool isPrimitiveReservedName(const std::string &Name) {
  return llvm::count(PrimitiveTypeNames, Name);
}

static inline bool isIntegralReservedName(const std::string &Name) {
  return llvm::count(IntegralTypeNames, Name);
}

static inline bool isVoidReservedName(const std::string &Name) {
  return Name == "void";
}

static inline bool isCReservedKeyword(const std::string &Name) {
  return llvm::count(CReservedKeywords, Name);
}

static llvm::cl::opt<uint64_t> ModelTypeIDSeed("model-type-id-seed",
                                               llvm::cl::desc("Set the seed "
                                                              "for "
                                                              "the generation "
                                                              "of "
                                                              "ID of model "
                                                              "Types"),
                                               llvm::cl::cat(MainCategory),
                                               llvm::cl::init(false));

class RNG {
  std::mt19937_64 generator;
  std::uniform_int_distribution<uint64_t> distribution;

public:
  RNG() :
    generator(ModelTypeIDSeed.getNumOccurrences() ? ModelTypeIDSeed.getValue() :
                                                    std::random_device()()),
    distribution(std::numeric_limits<uint64_t>::min(),
                 std::numeric_limits<uint64_t>::max()) {}

  uint64_t get() { return distribution(generator); }
};

llvm::ManagedStatic<RNG> IDGenerator;

model::Type::Type(TypeKind::Values TK, llvm::StringRef NameRef) :
  model::Type::Type(TK, IDGenerator->get(), NameRef) {
}

bool Type::verify() const {
  if (not ID)
    return false;
  if (Name.empty())
    return false;
  if (model::isCReservedKeyword(Name))
    return false;
  if (llvm::StringRef(Name).contains(' '))
    return false;
  return true;
}

bool Qualifier::verify() const {
  switch (Kind) {
  case QualifierKind::Invalid:
    return false;
  case QualifierKind::Pointer:
  case QualifierKind::Const:
    return not Size;
  case QualifierKind::Array:
    return Size;
  }
  return false;
}

bool QualifiedType::verify() const {

  if (not UnqualifiedType.Root)
    return false;

  model::Type *Underlying = UnqualifiedType.get();
  if (not Underlying or not model::verifyType(Underlying))
    return false;

  // This type has no qualifiers, so we're cool
  if (Qualifiers.empty())
    return true;

  auto QIt = Qualifiers.begin();
  auto QEnd = Qualifiers.end();
  for (; QIt != QEnd; ++QIt) {
    // Each qualifier needs to verify
    if (not QIt->verify())
      return false;

    auto NextQIt = std::next(QIt);
    // Check that we have not two consecutive const qualifiers
    if (NextQIt != QEnd and QIt->isConstQualifier()
        and NextQIt->isConstQualifier())
      return false;
  }
  return true;
}

bool PrimitiveType::verify() const {
  return Type::verify() and Kind == TypeKind::Primitive
         and isPrimitiveReservedName(Name);
}

bool EnumEntry::verify() const {
  return not Name.empty() and not Aliases.count(Name) and not Aliases.count("");
}

bool EnumType::verify() const {
  if (not Type::verify())
    return false;

  if (Kind != TypeKind::Enum)
    return false;

  if (not UnderlyingType.Root)
    return false;

  auto *Underlying = dyn_cast_or_null<PrimitiveType>(UnderlyingType.get());
  if (not Underlying or not Underlying->verify())
    return false;

  if (not model::isIntegralReservedName(Underlying->Name))
    return false;

  if (Entries.empty())
    return false;

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Entry : Entries) {

    if (not Entry.verify())
      return false;

    if (not Names.insert(Entry.Name).second)
      return false;
  }

  return true;
}

bool TypedefType::verify() const {
  if (not Type::verify())
    return false;

  if (Kind != TypeKind::Typedef)
    return false;

  if (not UnderlyingType.UnqualifiedType.Root)
    return false;

  model::Type *Underlying = UnderlyingType.UnqualifiedType.get();
  if (not Underlying or Underlying == this)
    return false;

  return model::verifyType(Underlying);
}

bool AggregateField::verify() const {
  return not Name.empty() and FieldType.verify();
}

static bool isOnlyConstQualified(const QualifiedType &QT) {

  revng_assert(QT.verify());

  if (QT.Qualifiers.empty() or QT.Qualifiers.size() > 1)
    return false;

  return QT.Qualifiers[0].isConstQualifier();
}

struct VoidConstResult {
  bool IsVoid;
  bool IsConst;
};

static VoidConstResult isVoidConst(const QualifiedType *QualType) {
  VoidConstResult Result{ /* IsVoid */ false, /* IsConst */ false };

  bool Done = false;
  while (not Done) {

    // If the argument type is qualified try to get the unqualified version.
    // Warning: we only skip const-qualifiers here, cause the other qualifiers
    // actually produce a different type.
    const Type *UnqualType = nullptr;
    if (model::isQualifiedType(*QualType)) {

      // If it has a non-const qualifier, it can never be void because it's a
      // pointer or array, so we can break out.
      if (not isOnlyConstQualified(*QualType)) {
        Done = true;
        continue;
      }

      // We know that it's const-qualified here, and it only has one
      // qualifier, hence we can skip the const-qualifier.
      Result.IsConst = true;
      if (not QualType->UnqualifiedType.Root)
        return Result;
    }

    UnqualType = QualType->UnqualifiedType.get();

    switch (UnqualType->Kind) {

    // If we still have a typedef in our way, unwrap it and keep looking.
    case TypeKind::Typedef: {
      QualType = &cast<TypedefType>(UnqualType)->UnderlyingType;
    } break;

    // If we have a primitive type, check the name, and we're done.
    case TypeKind::Primitive: {
      Result.IsVoid = isVoidReservedName(UnqualType->Name);
      Done = true;
    } break;

    // In all the other cases it's not void, break from the while.
    default: {
      Done = true;
    } break;
    }
  }
  return Result;
}

bool StructType::verify() const {
  if (not Type::verify())
    return false;

  if (Kind != TypeKind::Struct)
    return false;

  if (not Size)
    return false;

  // Empty structs are a GNU extension and have size 0
  if (Fields.empty())
    return false;

  llvm::SmallSet<llvm::StringRef, 8> Names;
  auto NumFields = Fields.size();
  auto FieldIt = Fields.begin();
  auto FieldEnd = Fields.end();
  for (; FieldIt != FieldEnd; ++FieldIt) {
    auto &Field = *FieldIt;

    if (not Field.FieldType.UnqualifiedType.Root)
      return false;

    auto *FldType = Field.FieldType.UnqualifiedType.get();
    if (not FldType or FldType == this)
      return false;

    if (not Field.verify())
      return false;

    if (Field.Offset >= Size)
      return false;

    auto FieldEndOffset = Field.Offset + model::typeSize(FldType);
    auto NextFieldIt = std::next(FieldIt);
    if (NextFieldIt != FieldEnd) {
      // If this field is not the last, check that it does not overlap with the
      // following field.
      if (FieldEndOffset > NextFieldIt->Offset)
        return false;
    } else if (FieldEndOffset > Size) {
      // Otherwise, if this field is the last, check that it's not larger than
      // size.
      return false;
    }

    if (isVoidConst(&Field.FieldType).IsVoid)
      return false;

    bool New = Names.insert(Field.Name).second;
    if (not New)
      return false;
  }
  return true;
}

model::UnionField::UnionField(const model::QualifiedType &QT,
                              llvm::StringRef NameRef) :
  model::AggregateField::AggregateField(QT, NameRef), ID(IDGenerator->get()) {
}

model::UnionField::UnionField(model::QualifiedType &&QT,
                              llvm::StringRef NameRef) :
  model::AggregateField::AggregateField(std::move(QT), NameRef),
  ID(IDGenerator->get()) {
}

bool UnionType::verify() const {
  if (not Type::verify())
    return false;

  if (Kind != TypeKind::Union)
    return false;

  // Empty structs are a GNU extension and have size 0
  if (Fields.empty())
    return false;

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Field : Fields) {

    if (not Field.FieldType.UnqualifiedType.Root)
      return false;

    auto *FldType = Field.FieldType.UnqualifiedType.get();
    if (not FldType or FldType == this)
      return false;

    if (not Field.verify())
      return false;

    if (not Names.insert(Field.Name).second)
      return false;

    if (isVoidConst(&Field.FieldType).IsVoid)
      return false;
  }
  return true;
}

bool ArgumentType::verify() const {
  return Type.verify();
}

bool FunctionPointerType::verify() const {
  if (not Type::verify())
    return false;

  if (Kind != TypeKind::FunctionPointer)
    return false;

  if (not ReturnType.UnqualifiedType.Root)
    return false;

  auto *Ret = ReturnType.UnqualifiedType.get();
  if (not Ret or Ret == this or not model::verifyType(Ret))
    return false;

  for (auto &Group : llvm::enumerate(ArgumentTypes)) {
    auto &ArgType = Group.value();
    uint64_t ArgPos = Group.index();

    if (ArgType.Pos != ArgPos)
      return false;

    if (not ArgType.Type.UnqualifiedType.Root)
      return false;

    auto *Arg = ArgType.Type.UnqualifiedType.get();
    if (not Arg or Arg == this)
      return false;

    if (not model::verifyType(Arg))
      return false;

    VoidConstResult VoidConst = isVoidConst(&ArgType.Type);
    if (VoidConst.IsVoid) {
      // If we have a void argument it must be the only one, and the function
      // cannot be vararg.
      if (ArgumentTypes.size() > 1)
        return false;

      // Cannot have const-qualified void as argument.
      if (VoidConst.IsConst)
        return false;
    }
  }

  return true;
}

bool verifyTypeSystem(const SortedVector<UpcastableType> &Types) {

  // All types on their own should verify
  for (auto &Type : Types)
    if (not model::verifyType(Type))
      return false;

  // FIXME: should also check that there are no loops in the type system and
  // that there are no duplicate names
  return true;
}

// TODO: this size should depend on the binary we are lifting.
static constexpr const inline uint64_t PointerSize = 8;

static RecursiveCoroutine<uint64_t> typeSizeImpl(const model::Type *T);

static RecursiveCoroutine<uint64_t>
typeSizeImpl(const model::QualifiedType &QT) {
  revng_assert(QT.verify());

  auto QIt = QT.Qualifiers.begin();
  auto QEnd = QT.Qualifiers.end();

  for (; QIt != QEnd; ++QIt) {

    auto &Q = *QIt;
    switch (Q.Kind) {

    case QualifierKind::Invalid:
      rc_return 0;

    case QualifierKind::Pointer:
      // If we find a pointer, we're done
      rc_return PointerSize;

    case QualifierKind::Array: {
      // The size is equal to (number of elements of the array) * (size of a
      // single element).
      const model::QualifiedType ArrayElem{ QT.UnqualifiedType,
                                            { std::next(QIt), QEnd } };
      rc_return Q.Size *model::typeSizeImpl(ArrayElem);
    }

    case QualifierKind::Const:
      // Do nothing, just skip over it
      ;
    }
  }
  rc_return 0;
}

static RecursiveCoroutine<uint64_t> typeSizeImpl(const model::Type *T) {
  revng_assert(T);
  revng_assert(verifyType(T));

  if (T->Kind == TypeKind::Primitive) {
    if (isVoidReservedName(T->Name))
      rc_return 0;

    llvm::StringRef NameRef = T->Name;
    NameRef.consume_back("_t");
    NameRef.consume_front("uint");
    NameRef.consume_front("int");
    NameRef.consume_front("float");
    NameRef.consume_front("bool");
    uint64_t BitSize = 0ULL;
    bool Err = NameRef.consumeInteger(0, BitSize);
    revng_assert(not Err);
    switch (BitSize) {
    case 8:
      rc_return 1;
    case 16:
      rc_return 2;
    case 32:
      rc_return 4;
    case 64:
      rc_return 8;
    case 128:
      rc_return 16;
    default:
      revng_abort();
    }
    rc_return 0;
  }

  switch (T->Kind) {

  case TypeKind::Enum: {
    auto *U = llvm::cast<EnumType>(T)->UnderlyingType.get();
    rc_return rc_recur model::typeSizeImpl(U);
  }

  case TypeKind::Typedef: {
    auto *Typedef = llvm::cast<TypedefType>(T);
    rc_return rc_recur model::typeSizeImpl(Typedef->UnderlyingType);
  }

  case TypeKind::Struct:
    rc_return llvm::cast<StructType>(T)->Size;

  case TypeKind::Union: {
    auto *U = llvm::cast<UnionType>(T);
    uint64_t Max = 0ULL;
    for (const auto &Field : U->Fields) {
      auto FieldSize = rc_recur model::typeSizeImpl(Field.FieldType);
      Max = std::max(Max, FieldSize);
    }
    rc_return Max;
  }

  case TypeKind::FunctionPointer: {
    rc_return PointerSize;
  }

  default:
    rc_return 0;
  }
  rc_return 0;
};

uint64_t typeSize(const model::Type *T) {
  return typeSizeImpl(T);
}

uint64_t typeSize(const model::QualifiedType &T) {
  return typeSizeImpl(T);
}

} // namespace model
