import os


def main():
    basedir = os.getcwd()
    for name in os.listdir('.'):
        os.chdir(basedir)
        if os.path.isdir(name):
            os.chdir(name)
            print(open('output').read())
            os.system('ase gui -b res.traj')
            print(name)


if __name__ == '__main__':
    main()
