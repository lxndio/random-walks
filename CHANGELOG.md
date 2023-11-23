# 0.1.1 (2023-XX-XX)

- `SimpleDynamicProgram` now takes a type for each field to apply a corresponding kernel
- `DynamicProgramBuilder` has been modified accordingly to support passing different kernels for different field types
- `DynamicProgramBuilder::kernels()` is now used for that instead of for passing multiple kernels to a `MultiDynamicProgram`
- Field probabilities are removed in favor of field types 
