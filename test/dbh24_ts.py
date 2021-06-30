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


def main():

    name = 'dbh24_r1'
    init = get_dbh24_system(dbh24.get_dbh24_initial_states(name))
    final = get_dbh24_system(dbh24.get_dbh24_final_states(name))
    ts = get_dbh24_system(dbh24.get_dbh24_tst(name))


if __name__ == '__main__':
    main()
