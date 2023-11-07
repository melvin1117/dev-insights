import sys
from boot_loader.boot import Boot

def run(module_code: str) -> int:
    """Runs the applications. Any application requirements related checks can be implemented here.
    Args:
        module_code (str): module code name
    """
    boot = Boot(module_code)
    boot.start()
    return 0

if __name__ == "__main__":
    print("Application started execution.", sys.argv[1])
    run(module_code = sys.argv[1]) # get the app code
