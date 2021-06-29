import os

from ase import Atoms
import ase.data.dbh24 as dbh24
import ase.gui.gui


def get_dbh24_system(systems):
    atoms = None
    if isinstance(systems, str):
        return dbh24.create_dbh24_system(systems)
    for s in systems:
        satoms = dbh24.create_dbh24_system(s)
        if atoms is None:
            atoms = satoms
        else:
            satoms.positions[:, 2] += 5
            atoms += satoms
    return atoms


def view_all():
    for name in dbh24.dbh24_reaction_list.keys():
        init = get_dbh24_system(dbh24.get_dbh24_initial_states(name))
        final = get_dbh24_system(dbh24.get_dbh24_final_states(name))
        ts = get_dbh24_system(dbh24.get_dbh24_tst(name))
        gui = ase.gui.gui.GUI([init, ts, final],
                              rotations='-90x, -90y',
                              show_bonds=True, expr='', )
        gui.window['show-labels'] = 1
        gui.show_labels()
        gui.run()


def write():
    basedir = os.getcwd()
    for name in dbh24.dbh24_reaction_list.keys():
        os.chdir(basedir)
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        init = get_dbh24_system(dbh24.get_dbh24_initial_states(name))
        final = get_dbh24_system(dbh24.get_dbh24_final_states(name))
        ts = get_dbh24_system(dbh24.get_dbh24_tst(name))
        init.write('1.xyz')
        final.write('2.xyz')
        ts.write('ts.xyz')


if __name__ == '__main__':
    write()
