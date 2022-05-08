def produce_pdb_ligand_with_confidence(coors, filepath):
    f = open(filepath + '.pdb', 'w')

    for i, c in enumerate(coors):
        # print(c)
        record_name = "{:<6}".format('ATOM')
        serial = '{:>5}'.format(str(i % 9999))
        sep0 = '{:>1}'.format(' ')
        atom = '{:>4}'.format(str(c[3]))
        altLoc = '{:<1}'.format(' ')
        resname = '{:>3}'.format('UNL')
        sep1 = '{:>1}'.format(' ')
        chainid = '{:>1}'.format(' ')
        resseq = '{:>4}'.format(str(i % 9999))
        icode = '{:>1}'.format(' ')
        sep2 = '{:>3}'.format(' ')
        x = '{:>8}'.format(str(round(float(c[0]), 3)))
        y = '{:>8}'.format(str(round(float(c[1]), 3)))
        z = '{:>8}'.format(str(round(float(c[2]), 3)))
        ocp = '{:>6}'.format(str('1.00'))
        tempfactor = '{:>6}'.format(str(1))
        sep3 = '{:>10}'.format(' ')
        element = '{:>2}'.format(str(c[3]))
        charge = '{:<2}'.format(' ')
        f.write(record_name + serial + sep0 + atom + altLoc + resname + sep1 + chainid + resseq + icode + sep2
                + x + y + z + ocp + tempfactor + sep3 + element + charge + '\n')
    f.close()

    return