import cProfile
import runpy

profiler = cProfile.Profile()
profiler.enable()

runpy.run_module("joule")

profiler.disable()

# Save the profiling stats to a file
profiler.dump_stats("profile.prof")
