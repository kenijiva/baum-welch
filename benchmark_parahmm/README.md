# Benchmark

We compare our implementation against the benchmark implementation available here: https://github.com/firebb/parahmm.

The following changes were made to their source code:
1. removed printf statements within the timed function
2. refactored `TestGenerator.py` to be able to run the code with python3.
3. We hook their original baum_welch function call and call the original function wrapped in the same timing loop that we use in our own performance measurement implementation to get comparable measurements.
