Review the uncomitted changes by running git status and proactively fix problems you find:

- Do we have any dead code? delete it.
- Did we add quality tests? Prefer, fewer, higher quality tests.
- Are there any glaring perf holes we can improve on without severly sacrificing readability?
- Can we use more modern abstractions for improved safety and readability?
- Is documentation still up to date with the code?
- Can we consolidate documentation into less files? Ideally a single README.md where possible.
- For all python related dependencies and runtime, make sure to use uv.

 