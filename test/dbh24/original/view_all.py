import os


def main():
    basedir = os.getcwd()
    for name in os.listdir('.'):
        os.chdir(basedir)
        if os.path.isdir(name):
            os.chdir(name)
            print(open('output').read())
            print(name, flush=True)
            os.system('ase gui -b -g "" -R 0z,-90x,-90y start.traj res.traj')


if __name__ == '__main__':
    main()
