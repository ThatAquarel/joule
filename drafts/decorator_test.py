
def ui_section(section_name, top_margin=True):
    def decorator(func):
        def wrapper():
            if top_margin:
                print("spacing")
                print("spacing")
                print("spacing")

            print(section_name)
            print()

            func()
        
        return wrapper
    return decorator


@ui_section("test section")
def section_1():
    print("section info")
    print("section info1")
    print("section info2")


@ui_section("test situation")
def section_2():
    print("section info")
    print("section info1")
    print("section info2")

section_1()
section_2()

