
import os


def main():
    basedir = os.getcwd()
    for name in os.listdir('.'):
        os.chdir(basedir)
        if os.path.isdir(name):
            os.chdir(name)
            # print(open('output').read())
            print(name, flush=True)
            for i in ['forward', 'backward']:
                os.chdir(i)
                # os.system(
                #     'ase gui -b -g "" -R 0z,-90x,-90y forward/res.traj')
                os.system(
                    '''ase convert -s res.traj 'pic-{:0>5}.png'  -e 'atoms.rotate(100, "y"); atoms.rotate(90, "z")' ''')
                os.system(
                    f"rm -f out.gif video.avi && ffmpeg -f image2 -i 'pic-%5d.png'  video.avi   && ffmpeg -i video.avi {name}-{i}.gif")
                os.system('rm -f *.avi *.png')
                os.chdir('..')

if __name__ == '__main__':
    main()
