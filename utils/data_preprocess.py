from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
import lmdb
import utils.dataset_collate_ignore_none as clfn
import rdkit.Chem as Chem

'''
Electron density value normalization  for XTB density
1.    val -= 0.03
2.    val < .00001:   val = 0.00001
3.    val > 100: val = 100
4.    val = ln(val)
normalize from [ln(0.00001), ln(100)] to [0,1]

x,y,z normalization
normalize from [-10,10] to [-1,1]
'''

atom_type_config = { 'C':0, 'N':1 , 'O':2, 'F':3, 'Cl':4, 'Br':5, 'I':6, 'S':7, 'P':8, 'H':9 }


def split_train_valid_test(dir_path):
    dir_name = os.path.basename(dir_path)
    dir_parent_path = os.path.dirname(dir_path)
    filenames_config_train_path = os.path.join(dir_parent_path, "{}_filenames_train.pkl".format(dir_name))
    filenames_config_valid_path = os.path.join(dir_parent_path, "{}_filenames_valid.pkl".format(dir_name))
    filenames_config_test_path = os.path.join(dir_parent_path, "{}_filenames_test.pkl".format(dir_name))
    if not os.path.exists(filenames_config_train_path):
        filenames = os.listdir(dir_path)
        n = len(filenames)
        test_n = n//10
        train_n = n - 2*(n//10)
        with open(filenames_config_train_path, 'wb') as f:
            pickle.dump(filenames[:train_n], f)
        with open(filenames_config_valid_path, 'wb') as f:
            pickle.dump(filenames[train_n: n - test_n], f)
        with open(filenames_config_test_path, 'wb') as f:
            pickle.dump(filenames[n - test_n: ], f)
    return filenames_config_train_path, filenames_config_valid_path, filenames_config_test_path

def get_atom_coords(pdb_path):
    mol = Chem.rdmolfiles.MolFromPDBFile(pdb_path)
    conf = mol.GetConformer()
    atoms = mol.GetAtoms()
    result = []
    for atom in atoms:
        idx = atom.GetIdx()
        atom_str = atom.GetSymbol()
        coord = conf.GetAtomPosition(idx)
        result.append([coord.x, coord.y, coord.z, atom_type_config[atom_str]])
    return np.array(result,dtype=np.float32)

class ElectronDensityDirDataset(Dataset):
    def __init__(self, dir_path, split='train', sample_npoints = 20000, lowerbound_npoints = 40000, max_atom_count = 50):
        self.dir_path = dir_path
        self.dir_name = os.path.basename(dir_path)
        self.dir_parent_path = os.path.dirname(dir_path)
        self.sample_npoints = sample_npoints
        self.lowerbound_npoints =lowerbound_npoints
        self.max_atom_count = max_atom_count
        train_config_path, valid_config_path, test_config_path = split_train_valid_test(dir_path)
        if split == 'train':
            self.filenames_config_path = train_config_path
        elif split == 'valid':
            self.filenames_config_path = valid_config_path
        elif split == 'test':
            self.filenames_config_path = test_config_path
        else:
            raise Exception("illegal split name: {}".format(split))
        with open(self.filenames_config_path, 'rb') as f:
            filenames = pickle.load(f)
        self.filenames = filenames

    def _data_normalize_eletron_density(self, electron_density, npoints=None ):
        if npoints is not None:
            inds = np.random.randint(0, electron_density.shape[0], npoints)
            electron_density = electron_density[inds]
        electron_density[:, -1] -= 0.03
        electron_density[:, -1] = np.max(electron_density[:, np.newaxis, -1], initial=1e-5, axis=-1)
        electron_density[:, -1] = np.min(electron_density[:, np.newaxis, -1], initial=100, axis=-1)
        electron_density[:, -1] = np.log(electron_density[:, -1])
        electron_density[:, -1] = (electron_density[:, -1] - np.log(1e-5)) / (np.log(100) - np.log(1e-5))
        electron_density[:, -1] += 1e-5
        electron_density[:, 0:-1] /= 10
        return electron_density

    def _data_normalize_coord_atoms(self, coord_atoms, max_atom_count):
        coord_atoms[:,0:-1] /= 10 #coordinates normalized by divide 10
        result = np.pad(coord_atoms, ((0, max_atom_count - coord_atoms.shape[0]), (0, 0)), 'constant', constant_values=(0, 100))
        return result

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        fpath = os.path.join(self.dir_path, filename)
        electron_density = np.load(fpath).astype(np.float32)
        pdb_path = os.path.join("{}_pdb".format(self.dir_path), "{}.pdb".format(filename[:-4]))
        coord_atoms = get_atom_coords(pdb_path)

        if electron_density.shape[0] < self.lowerbound_npoints or coord_atoms.shape[0] > self.max_atom_count:
            return None, None, None
        electron_density = self._data_normalize_eletron_density(electron_density, self.sample_npoints)
        coord_atoms = self._data_normalize_coord_atoms(coord_atoms, self.max_atom_count)
        return filename, electron_density, coord_atoms

    def __len__(self) -> int:
        return len(self.filenames)

def test_ElectronDensityDirDataset():
    dataset = ElectronDensityDirDataset("/home/jovyan/data/xtb_ed", split='valid')
    data_loader = DataLoader(dataset, num_workers = 0, collate_fn = clfn.collate_ignore_none,  batch_size=2)
    for idx, data in enumerate(data_loader):
        filename, electron_density, coord_atoms = data
        print("{}: {}".format(filename, electron_density.shape))
        # convert_npy_to_pdb(electron_density[0], "{}_ed_2w".format(filename[0]))
        if idx % 5000 == 0:
            print("[%d/%d]" % (idx, len(data_loader)))

def data_to_lmdb(dir_path, suffix="train", write_frequency=5000):
    dataset = ElectronDensityDirDataset(dir_path)
    data_loader = DataLoader(dataset, num_workers=0)
    dir_name = os.path.basename(dir_path)
    dir_parent_path = os.path.dirname(dir_path)
    lmdb_path = os.path.join(dir_parent_path, "{}_{}.lmdb".format(dir_name, suffix))
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    keys = []
    for idx, data in enumerate(data_loader):
        filename, electron_density = data
        txn.put(u'{}'.format(filename[0]).encode('ascii'), pickle.dumps(electron_density[0]))
        keys.append(u'{}'.format(filename[0]).encode('ascii'))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__ == '__main__':
    # data_to_lmdb("/home/jovyan/data/xtb_ed")
    test_ElectronDensityDirDataset()

# print(type(data), data)



