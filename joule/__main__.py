from joule.app import run


if __name__ == "__main__":
    # application entrypoint
    run()

    # for optimizations
    #
    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()
    # ...
    # profiler.disable()
    # profiler.dump_stats("profile.prof")
