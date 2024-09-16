#this script it's here just for developmenet purposes
#run it everytime you make changes to the library file
import subprocess

def reinstall_package():
    # Uninstall the package
    subprocess.run(["pip", "uninstall", "-y", "atpp"], check=True)
    # Install the package
    subprocess.run(["pip", "install", "."], check=True)

if __name__ == "__main__":
    reinstall_package()