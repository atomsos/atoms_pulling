
import ase.io

def get_index(index_file, name):
    content = open(index_file).read()
    idx = content.find('[ ' + name + ' ]\n')
    if idx == -1:
        return None
    content = content[idx:]
    content = content[content.index(']')+1:].strip()
    if '\n[' in content:
        content = content[:content.find('\n[')]
    return list(map(lambda x: int(x) - 1, content.strip().split()))


if __name__ == '__main__':
    indexs = get_index('index.ndx', 'protein_lig')
    traj = ase.io.read('traj.traj', index=':')
    traj = [_[indexs] for _ in traj]
    ase.io.write('traj_idx.traj', traj)