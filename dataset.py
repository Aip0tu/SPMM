from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import os
import math
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
import pickle
from rdkit import RDLogger
from calc_property import calculate_property
try:
    from pysmilesutils.augment import MolAugmenter
except ImportError:
    MolAugmenter = None
RDLogger.DisableLog('rdApp.*')

FLUODB_RDKIT_DESCRIPTOR_COUNT = 200
FLUODB_RDKIT_DESCRIPTOR_ITEMS = [
    item for item in Descriptors.descList if item[0] != 'Ipc'
][:FLUODB_RDKIT_DESCRIPTOR_COUNT]
FLUODB_RDKIT_DESCRIPTOR_NAMES = [name for name, _ in FLUODB_RDKIT_DESCRIPTOR_ITEMS]
FLUODB_RDKIT_DESCRIPTOR_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(
    FLUODB_RDKIT_DESCRIPTOR_NAMES
)


class SMILESDataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        if data_length is not None:
            with open(data_path, 'r') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    lines.append(f.readline())
        else:
            with open(data_path, 'r') as f:
                lines = f.readlines()
        self.data = [l.strip() for l in lines]
        with open('./normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - self.property_mean) / self.property_std

        return properties, '[CLS]' + smiles


class SMILESDataset_BACEC(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['mol']), isomericSmiles=False, canonical=True)
        value = int(self.data[index]['Class'])

        return '[CLS]' + smiles, value


class SMILESDataset_BACER(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(6.420878294545455)
        self.value_std = torch.tensor(1.345219669175284)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor(self.data[index]['target'].item())
        return '[CLS]' + smiles, value


class SMILESDataset_LIPO(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(2.162904761904762)
        self.value_std = torch.tensor(1.210992810122257)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['exp'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_Clearance(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(51.503692077727955)
        self.value_std = torch.tensor(53.50834365711207)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['target'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_BBBP(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data)) if Chem.MolFromSmiles(data.iloc[i]['smiles'])]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        label = int(self.data[index]['p_np'])

        return '[CLS]' + smiles, label


class SMILESDataset_ESOL(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(-2.8668758314855878)
        self.value_std = torch.tensor(2.066724108076815)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['ESOL predicted log solubility in mols per litre'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_Freesolv(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(-3.2594736842105267)
        self.value_std = torch.tensor(3.2775297233608893)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = (self.data[index]['target'] - self.value_mean) / self.value_std

        return '[CLS]' + smiles, value


class SMILESDataset_Clintox(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output = 2

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor([float(self.data[index]['FDA_APPROVED']), float(self.data[index]['CT_TOX'])])

        return '[CLS]' + smiles, value


class SMILESDataset_SIDER(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output = 27

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False, canonical=True, kekuleSmiles=False)
        value = self.data[index].values.tolist()[1:]
        value = torch.tensor([i.item() for i in value])
        return '[CLS]' + smiles, value


class SMILESDataset_DILI(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['Smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['Liver'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_FluoDB(Dataset):
    def __init__(self, data_path, target_name=None, value_mean=None, value_std=None, shuffle=False):
        data = pd.read_csv(data_path)
        if target_name is None:
            target_columns = [c for c in data.columns if c not in ['smiles', 'solvent']]
            if len(target_columns) != 1:
                raise ValueError(f'Cannot infer target column from {data_path}.')
            target_name = target_columns[0]

        self.target_name = target_name
        self.data = []
        for i in range(len(data)):
            row = data.iloc[i]
            if pd.isna(row['smiles']) or pd.isna(row['solvent']) or pd.isna(row[target_name]):
                continue

            mol = Chem.MolFromSmiles(row['smiles'])
            solvent_mol = Chem.MolFromSmiles(row['solvent'])
            if mol is None or solvent_mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            solvent = Chem.MolToSmiles(solvent_mol, isomericSmiles=False, canonical=True)
            self.data.append((smiles, solvent, float(row[target_name])))

        if not self.data:
            raise ValueError(f'No valid samples found in {data_path}.')

        values = torch.tensor([row[2] for row in self.data], dtype=torch.float)
        self.value_mean = torch.as_tensor(values.mean() if value_mean is None else value_mean, dtype=torch.float)
        self.value_std = torch.as_tensor(values.std(unbiased=False) if value_std is None else value_std, dtype=torch.float)
        if self.value_std.item() == 0:
            self.value_std = torch.tensor(1.0)

        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles, solvent, value = self.data[index]
        value = (torch.tensor(value, dtype=torch.float) - self.value_mean) / self.value_std
        return '[CLS]' + smiles, '[CLS]' + solvent, value


class SMILESDataset_FluoDB_MultiTask(Dataset):
    target_names = ['abs', 'emi', 'plqy', 'e']
    rdkit_descriptor_count = FLUODB_RDKIT_DESCRIPTOR_COUNT
    rdkit_descriptor_names = FLUODB_RDKIT_DESCRIPTOR_NAMES
    rdkit_descriptor_calculator = FLUODB_RDKIT_DESCRIPTOR_CALCULATOR
    fingerprint_n_bits = 512
    _scaffold_split_cache = {}
    _rdkit_descriptor_cache = {}
    _fingerprint_cache = {}
    lite_target_columns = {
        'abs': 'absorption/nm',
        'emi': 'emission/nm',
        'plqy': 'plqy',
        'e': 'e/m-1cm-1',
    }
    solvent_descriptor_columns = [
        'dielectric_constant',
        'polarity',
        'refractive_index',
        'hbond_donor',
        'hbond_acceptor',
        'ET30',
    ]
    target_transform_config = {
        'plqy': 'logit',
        'e': 'log10',
    }
    plqy_transform_eps = 1e-4
    e_transform_eps = 1e-12

    def __init__(self, data_dir, split='train', value_mean=None, value_std=None,
                 solvent_desc_mean=None, solvent_desc_std=None, solvent_descriptor_file=None,
                 mol_desc_mean=None, mol_desc_std=None, split_strategy='existing',
                 split_seed=42, split_ratios=(0.8, 0.1, 0.1), shuffle=False):
        self.split = split
        self.data_dir = data_dir
        self.solvent_descriptor_file = solvent_descriptor_file
        self.split_strategy = split_strategy
        data = self._load_split(data_dir, split, split_strategy, split_seed, split_ratios)
        solvent_descriptor_table = self._load_solvent_descriptor_table(solvent_descriptor_file)

        self.data = []
        values = []
        masks = []
        solvent_descs = []
        mol_descs = []
        mol_fps = []
        mol_feature_cache = {}
        solvent_desc_cache = {}
        for i in range(len(data)):
            row = data.iloc[i]
            if pd.isna(row['smiles']) or pd.isna(row['solvent']):
                continue

            mol = Chem.MolFromSmiles(row['smiles'])
            solvent_mol = Chem.MolFromSmiles(row['solvent'])
            if mol is None or solvent_mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            solvent = Chem.MolToSmiles(solvent_mol, isomericSmiles=False, canonical=True)
            target_values, target_mask = self._extract_targets(row)
            if not any(target_mask):
                continue

            if solvent not in solvent_desc_cache:
                solvent_desc_cache[solvent] = self._build_solvent_descriptor(
                    solvent_mol,
                    solvent,
                    solvent_descriptor_table,
                )
            if smiles not in mol_feature_cache:
                mol_feature_cache[smiles] = (
                    self._build_rdkit_descriptor(mol),
                    self._build_morgan_fingerprint(mol),
                )
            solvent_desc = solvent_desc_cache[solvent]
            mol_desc, mol_fp = mol_feature_cache[smiles]
            self.data.append((smiles, solvent))
            values.append(target_values)
            masks.append(target_mask)
            solvent_descs.append(solvent_desc)
            mol_descs.append(mol_desc)
            mol_fps.append(mol_fp)

        if not self.data:
            raise ValueError(f'No valid FluoDB multitask samples found for split "{split}" in {data_dir}.')

        raw_values = torch.tensor(values, dtype=torch.float)
        target_masks = torch.tensor(masks, dtype=torch.bool)
        transformed_values = self.transform_targets(raw_values)
        raw_solvent_desc = torch.tensor(solvent_descs, dtype=torch.float)
        raw_mol_desc = torch.tensor(mol_descs, dtype=torch.float)
        raw_mol_fp = torch.tensor(mol_fps, dtype=torch.float)

        if value_mean is None or value_std is None:
            means, stds = [], []
            for target_idx in range(len(self.target_names)):
                observed = transformed_values[target_masks[:, target_idx], target_idx]
                if observed.numel() == 0:
                    means.append(torch.tensor(0.0))
                    stds.append(torch.tensor(1.0))
                else:
                    means.append(observed.mean())
                    std = observed.std(unbiased=False)
                    stds.append(std if std.item() != 0 else torch.tensor(1.0))
            self.value_mean = torch.stack(means).float()
            self.value_std = torch.stack(stds).float()
        else:
            self.value_mean = torch.as_tensor(value_mean, dtype=torch.float)
            self.value_std = torch.as_tensor(value_std, dtype=torch.float)

        if solvent_desc_mean is None or solvent_desc_std is None:
            self.solvent_desc_mean = raw_solvent_desc.mean(dim=0)
            self.solvent_desc_std = raw_solvent_desc.std(dim=0, unbiased=False)
            self.solvent_desc_std[self.solvent_desc_std == 0] = 1.0
        else:
            self.solvent_desc_mean = torch.as_tensor(solvent_desc_mean, dtype=torch.float)
            self.solvent_desc_std = torch.as_tensor(solvent_desc_std, dtype=torch.float)

        if mol_desc_mean is None or mol_desc_std is None:
            self.mol_desc_mean = raw_mol_desc.mean(dim=0)
            self.mol_desc_std = raw_mol_desc.std(dim=0, unbiased=False)
            self.mol_desc_std[self.mol_desc_std == 0] = 1.0
        else:
            self.mol_desc_mean = torch.as_tensor(mol_desc_mean, dtype=torch.float)
            self.mol_desc_std = torch.as_tensor(mol_desc_std, dtype=torch.float)

        normalized_values = (transformed_values - self.value_mean) / self.value_std
        normalized_values = torch.where(target_masks, normalized_values, torch.zeros_like(normalized_values))
        normalized_solvent_desc = (raw_solvent_desc - self.solvent_desc_mean) / self.solvent_desc_std
        normalized_mol_desc = (raw_mol_desc - self.mol_desc_mean) / self.mol_desc_std

        self.values = normalized_values
        self.target_masks = target_masks.float()
        self.solvent_descs = normalized_solvent_desc
        self.mol_descs = normalized_mol_desc
        self.mol_fps = raw_mol_fp

        if shuffle:
            order = list(range(len(self.data)))
            random.shuffle(order)
            self.data = [self.data[i] for i in order]
            self.values = self.values[order]
            self.target_masks = self.target_masks[order]
            self.solvent_descs = self.solvent_descs[order]
            self.mol_descs = self.mol_descs[order]
            self.mol_fps = self.mol_fps[order]

    def _load_split(self, data_dir, split, split_strategy='existing', split_seed=42, split_ratios=(0.8, 0.1, 0.1)):
        if split_strategy == 'scaffold':
            split_key = 'valid' if split == 'val' else split
            ratios = tuple(float(r) for r in split_ratios)
            cache_key = (os.path.abspath(data_dir), int(split_seed), ratios)
            if cache_key not in self._scaffold_split_cache:
                data = self._load_all_data(data_dir)
                self._scaffold_split_cache[cache_key] = self._build_scaffold_split_map(
                    data,
                    split_seed,
                    ratios,
                )
            if split_key not in self._scaffold_split_cache[cache_key]:
                raise ValueError(f'Unknown split: {split}')
            return self._scaffold_split_cache[cache_key][split_key].copy()
        if split_strategy != 'existing':
            raise ValueError(f'Unknown split_strategy: {split_strategy}')

        lite_path = os.path.join(data_dir, 'FluoDB-Lite.csv')
        if os.path.exists(lite_path):
            data = pd.read_csv(lite_path)
            if 'split' not in data.columns:
                raise ValueError(f'{lite_path} must contain a split column or use split_strategy="scaffold".')
            data = data[data['split'].astype(str).str.lower() == split.lower()].copy()
            data = data.rename(columns={v: k for k, v in self.lite_target_columns.items()})
            return data[['smiles', 'solvent'] + self.target_names]

        merged = None
        for target in self.target_names:
            path = os.path.join(data_dir, f'{target}_{split}.csv')
            if not os.path.exists(path):
                continue
            data = pd.read_csv(path)
            if target not in data.columns:
                raise ValueError(f'{path} must contain a {target} column.')
            data = data[['smiles', 'solvent', target]]
            merged = data if merged is None else merged.merge(data, on=['smiles', 'solvent'], how='outer')

        if merged is None:
            raise FileNotFoundError(f'Cannot find FluoDB-Lite.csv or per-target {split} CSV files in {data_dir}.')
        return merged

    def _load_all_data(self, data_dir):
        lite_path = os.path.join(data_dir, 'FluoDB-Lite.csv')
        if os.path.exists(lite_path):
            data = pd.read_csv(lite_path)
            data = data.rename(columns={v: k for k, v in self.lite_target_columns.items()})
            return data[['smiles', 'solvent'] + self.target_names]

        merged = None
        for target in self.target_names:
            frames = []
            for split in ['train', 'valid', 'test']:
                path = os.path.join(data_dir, f'{target}_{split}.csv')
                if not os.path.exists(path):
                    continue
                data = pd.read_csv(path)
                if target not in data.columns:
                    raise ValueError(f'{path} must contain a {target} column.')
                frames.append(data[['smiles', 'solvent', target]])
            if not frames:
                continue
            target_data = pd.concat(frames, ignore_index=True)
            target_data = target_data.groupby(['smiles', 'solvent'], as_index=False)[target].mean()
            merged = target_data if merged is None else merged.merge(target_data, on=['smiles', 'solvent'], how='outer')

        if merged is None:
            raise FileNotFoundError(f'Cannot find FluoDB-Lite.csv or per-target CSV files in {data_dir}.')
        return merged

    def _apply_scaffold_split(self, data, split, split_seed, split_ratios):
        split = 'valid' if split == 'val' else split
        split_map = self._build_scaffold_split_map(data, split_seed, split_ratios)
        if split not in split_map:
            raise ValueError(f'Unknown split: {split}')
        return split_map[split].copy()

    def _build_scaffold_split_map(self, data, split_seed, split_ratios):
        ratios = [float(r) for r in split_ratios]
        if len(ratios) != 3 or sum(ratios) <= 0:
            raise ValueError('split_ratios must contain three positive values.')
        ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]

        groups = {}
        for idx, row in data.iterrows():
            if pd.isna(row['smiles']):
                continue
            scaffold = self._scaffold_key(row['smiles'])
            groups.setdefault(scaffold, []).append(idx)

        grouped = list(groups.items())
        rng = random.Random(split_seed)
        rng.shuffle(grouped)
        grouped.sort(key=lambda item: len(item[1]), reverse=True)

        names = ['train', 'valid', 'test']
        desired = {
            'train': int(round(len(data) * ratios[0])),
            'valid': int(round(len(data) * ratios[1])),
        }
        desired['test'] = max(0, len(data) - desired['train'] - desired['valid'])
        split_indices = {name: [] for name in names}

        for _, indices in grouped:
            target_split = max(
                names,
                key=lambda name: desired[name] - len(split_indices[name]),
            )
            split_indices[target_split].extend(indices)

        return {
            name: data.loc[sorted(indices)].reset_index(drop=True)
            for name, indices in split_indices.items()
        }

    @staticmethod
    def _scaffold_key(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return str(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold:
            return scaffold
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

    def _load_solvent_descriptor_table(self, solvent_descriptor_file):
        if not solvent_descriptor_file:
            return {}
        table = pd.read_csv(solvent_descriptor_file)
        if 'solvent' not in table.columns:
            raise ValueError(f'{solvent_descriptor_file} must contain a solvent column.')

        descriptor_columns = [c for c in self.solvent_descriptor_columns if c in table.columns]
        if not descriptor_columns:
            raise ValueError(f'{solvent_descriptor_file} must contain at least one known solvent descriptor column.')

        lookup = {}
        for _, row in table.iterrows():
            mol = Chem.MolFromSmiles(row['solvent'])
            if mol is None:
                continue
            solvent = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            lookup[solvent] = [float(row[c]) if not pd.isna(row[c]) else 0.0 for c in descriptor_columns]
        return {'columns': descriptor_columns, 'values': lookup}

    def _extract_targets(self, row):
        target_values = []
        target_mask = []
        for target in self.target_names:
            value = row[target] if target in row else float('nan')
            if pd.isna(value):
                target_values.append(0.0)
                target_mask.append(False)
            else:
                target_values.append(float(value))
                target_mask.append(True)
        return target_values, target_mask

    @classmethod
    def transform_targets(cls, values):
        transformed = values.clone()
        was_1d = transformed.dim() == 1
        if was_1d:
            transformed = transformed.unsqueeze(0)

        if 'plqy' in cls.target_names:
            target_idx = cls.target_names.index('plqy')
            plqy = transformed[:, target_idx].clamp(cls.plqy_transform_eps, 1.0 - cls.plqy_transform_eps)
            transformed[:, target_idx] = torch.log(plqy / (1.0 - plqy))

        if 'e' in cls.target_names:
            target_idx = cls.target_names.index('e')
            e_value = transformed[:, target_idx].clamp_min(cls.e_transform_eps)
            transformed[:, target_idx] = torch.log10(e_value)

        return transformed.squeeze(0) if was_1d else transformed

    @classmethod
    def inverse_transform_targets(cls, values):
        restored = values.clone()
        was_1d = restored.dim() == 1
        if was_1d:
            restored = restored.unsqueeze(0)

        if 'plqy' in cls.target_names:
            target_idx = cls.target_names.index('plqy')
            restored[:, target_idx] = torch.sigmoid(restored[:, target_idx]).clamp(0.0, 1.0)

        if 'e' in cls.target_names:
            target_idx = cls.target_names.index('e')
            base = torch.tensor(10.0, dtype=restored.dtype, device=restored.device)
            restored[:, target_idx] = torch.pow(base, restored[:, target_idx])

        return restored.squeeze(0) if was_1d else restored

    @classmethod
    def _build_rdkit_descriptor(cls, mol):
        key = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        if key in cls._rdkit_descriptor_cache:
            return list(cls._rdkit_descriptor_cache[key])

        desc = []
        try:
            values = cls.rdkit_descriptor_calculator.CalcDescriptors(mol)
        except Exception:
            values = [0.0] * cls.rdkit_descriptor_count
        for value in values:
            value = float(value)
            if not math.isfinite(value):
                value = 0.0
            desc.append(value)
        cls._rdkit_descriptor_cache[key] = tuple(desc)
        return desc

    @classmethod
    def _build_morgan_fingerprint(cls, mol):
        key = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        if key in cls._fingerprint_cache:
            return list(cls._fingerprint_cache[key])

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=cls.fingerprint_n_bits)
        bits = [float(bit) for bit in fp.ToBitString()]
        cls._fingerprint_cache[key] = tuple(bits)
        return bits

    def _build_solvent_descriptor(self, solvent_mol, solvent, solvent_descriptor_table):
        rdkit_desc = self._build_rdkit_descriptor(solvent_mol)
        if not solvent_descriptor_table:
            return rdkit_desc

        extra_values = solvent_descriptor_table['values'].get(
            solvent,
            [0.0] * len(solvent_descriptor_table['columns'])
        )
        return rdkit_desc + extra_values

    @property
    def solvent_desc_dim(self):
        return int(self.solvent_descs.size(1))

    @property
    def mol_desc_dim(self):
        return int(self.mol_descs.size(1))

    @property
    def mol_fp_dim(self):
        return int(self.mol_fps.size(1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles, solvent = self.data[index]
        return (
            '[CLS]' + smiles,
            '[CLS]' + solvent,
            self.mol_descs[index],
            self.mol_fps[index],
            self.solvent_descs[index],
            self.values[index],
            self.target_masks[index],
        )


class SMILESDataset_USPTO(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, aug=False):
        self.is_aug = aug
        if self.is_aug and MolAugmenter is None:
            raise ImportError('pysmilesutils is required for USPTO augmentation.')
        self.aug = MolAugmenter() if MolAugmenter is not None else None
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.data = [line.strip() for line in lines]

        if shuffle:
            random.shuffle(self.data)
        if data_length:
            self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rs, ps = self.data[index].split('\t')
        if self.is_aug and random.random() > 0.5:
            r_mol = self.aug([Chem.MolFromSmiles(rs[:])])[0]
            rs = Chem.MolToSmiles(r_mol, canonical=False, isomericSmiles=False)
            p_mol = self.aug([Chem.MolFromSmiles(ps[:])])[0]
            ps = Chem.MolToSmiles(p_mol, canonical=False, isomericSmiles=False)
        return '[CLS]' + rs, '[CLS]' + ps


class SMILESDataset_USPTO_reverse(Dataset):
    def __init__(self, data_length=None, shuffle=False, mode=None, aug=False):
        with open('./data/6_RXNprediction/USPTO-50k/uspto_50.pickle', 'rb') as f:
            data = pickle.load(f)
        data = [data.iloc[i] for i in range(len(data))]
        self.data = [d for d in data if d['set'] == mode]
        self.is_aug = aug
        if self.is_aug and MolAugmenter is None:
            raise ImportError('pysmilesutils is required for USPTO augmentation.')
        self.aug = MolAugmenter() if MolAugmenter is not None else None

        if shuffle:
            random.shuffle(self.data)
        if data_length:
            self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        # r_type = d['reaction_type']  # 如需反应类型可在此读取
        p_mol = d['products_mol']
        r_mol = d['reactants_mol']
        do_aug = self.is_aug and random.random() > 0.5
        if do_aug:
            p_mol = self.aug([p_mol])[0]
            r_mol = self.aug([r_mol])[0]
        return '[CLS]' + Chem.MolToSmiles(p_mol, canonical=not do_aug, isomericSmiles=False), \
               '[CLS]' + Chem.MolToSmiles(r_mol, canonical=not do_aug, isomericSmiles=False)
