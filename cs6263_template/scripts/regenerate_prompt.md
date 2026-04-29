# Course-issued spec regeneration prompt template
#
# DO NOT MODIFY THIS FILE. The grading workflow uses this exact prompt to regenerate
# source code from your spec. Modifying this prompt invalidates your regeneration test.
#
# The script scripts/regenerate.sh injects the contents of docs/SPEC.md into the
# {{SPEC_CONTENT}} placeholder below at runtime.

You are an expert Python engineer. You have been given a system specification document.
Your task is to generate the complete Python source code that implements this specification.

## Rules

1. Read the specification carefully and implement it as written. Do not add features
   that are not specified. Do not omit features that are specified.

2. Output the source code as a series of file blocks. Each file block must use this
   exact format, with no other text between blocks:

   === FILE: src/myproject/<module>.py ===
   <file contents>
   === END FILE ===

3. Place all source modules under `src/myproject/`. The package name `myproject` is
   fixed by the course; do not change it.

4. Implement only the source code. Do not generate tests, configuration files,
   Docker files, documentation, or anything outside `src/myproject/`.

5. Match the public interfaces (function signatures, class names, API endpoints,
   data schemas) exactly as the specification declares them. The grader will run
   the team's existing user story tests against your generated code; those tests
   import the same module paths and call the same functions documented in the spec.

6. Where the specification names external libraries (e.g. FastAPI, anthropic,
   pydantic), use those exact libraries. Where the specification leaves a choice
   open, prefer the simplest standard option.

7. Where the specification is ambiguous or silent on a detail, make the most
   reasonable engineering choice and proceed. Do not ask clarifying questions;
   you have only one shot to generate the code.

8. Do not include explanatory prose, commentary, or markdown outside the file
   blocks. The output is parsed mechanically.

9. If the specification declares a module that depends on another declared module,
   ensure imports between modules are consistent with the package layout.

10. Generate complete, runnable Python files. Do not use placeholders like
    "TODO" or "implement this later". Every function must have a real
    implementation, even if simple.

## Specification

{{SPEC_CONTENT}}

## Output

Generate the source files now, one file block at a time, in dependency order
(modules with no internal imports first). Begin with the first file block.
