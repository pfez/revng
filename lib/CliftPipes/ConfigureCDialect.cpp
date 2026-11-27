//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftPipes/ConfigureCDialect.h"
#include "revng/Support/YAMLTraits.h"

using namespace clift;

namespace {

/// The dynamic configuration of the pipe. The options it does not mention keep
/// the default of the pipeline, which is the most permissive dialect.
struct CDialectConfiguration {
  CDialect Dialect = CDialect::permissive();
};

} // namespace

template<>
struct llvm::yaml::MappingTraits<CDialectConfiguration> {
  static void mapping(IO &TheIO, CDialectConfiguration &Configuration) {
    for (const CDialectOption &Option : CDialectOptions) {
      TheIO.mapOptional(Option.ConfigurationKey.data(),
                        Configuration.Dialect.*Option.Member);
    }
  }
};

static CDialect parseCDialect(llvm::StringRef DynamicConfig) {
  if (DynamicConfig.empty())
    return CDialect::permissive();

  auto Configuration = fromString<CDialectConfiguration>(DynamicConfig);
  return llvm::cantFail(std::move(Configuration)).Dialect;
}

using Pipe = revng::pypeline::piperuns::ConfigureFunctionCDialect;

Pipe::ConfigureFunctionCDialect(const Model &Model,
                                llvm::StringRef Config,
                                llvm::StringRef DynamicConfig,
                                CliftFunctionContainer &ModuleContainer) :
  CliftFunctionMixin(ModuleContainer), Dialect(parseCDialect(DynamicConfig)) {
}

void Pipe::runOnCliftFunction(const model::Function &Model, FunctionOp Op) {
  if (Dialect != CDialect::strict())
    setCDialect(Op->getParentOfType<mlir::ModuleOp>(), Dialect);
}
