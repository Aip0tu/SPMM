import argparse
import ast
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from d_fluodb_regression import SPMM_fluodb_regressor, build_tokenizer


EXAMPLE_FLUOROPHORES = [
    "Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1",
    "Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1",
]

EXAMPLE_SOLVENTS = [
    "ClCCl",
    "O",
]

TWO_CHAR_ELEMENTS = {
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "Ba", "Be", "Bh", "Bi",
    "Bk", "Br", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Cn", "Co", "Cr", "Cs",
    "Cu", "Db", "Ds", "Dy", "Er", "Es", "Eu", "Fe", "Fl", "Fm", "Fr", "Ga",
    "Gd", "Ge", "He", "Hf", "Hg", "Ho", "Hs", "In", "Ir", "Kr", "La", "Li",
    "Lr", "Lu", "Lv", "Mc", "Mg", "Mn", "Mo", "Mt", "Na", "Nb", "Nd", "Ne",
    "Nh", "Ni", "No", "Np", "Og", "Os", "Pb", "Pd", "Pm", "Po", "Pr", "Pt",
    "Pu", "Ra", "Rb", "Re", "Rf", "Rg", "Rh", "Rn", "Ru", "Sb", "Sc", "Se",
    "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl",
    "Tm", "Ts", "Xe", "Yb", "Zn", "Zr",
}

AROMATIC_ATOMS = {"b", "c", "n", "o", "p", "s"}


def safe_file_stem(text: str, max_len: int = 80) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    if not stem:
        stem = "molecule"
    return stem[:max_len]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def canonicalize_smiles(smiles: str) -> Tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return mol, canonical


def scan_smiles_atom_spans(smiles: str) -> List[Tuple[int, int]]:
    spans = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            end = smiles.find("]", i + 1)
            if end < 0:
                raise ValueError(f"Unclosed bracket atom in SMILES: {smiles}")
            spans.append((i, end + 1))
            i = end + 1
            continue

        if ch.isupper():
            end = i + 1
            if i + 1 < len(smiles) and smiles[i:i + 2] in TWO_CHAR_ELEMENTS:
                end = i + 2
            spans.append((i, end))
            i = end
            continue

        if ch in AROMATIC_ATOMS or ch == "*":
            spans.append((i, i + 1))
            i += 1
            continue

        i += 1

    return spans


def canonical_atom_spans(mol: Chem.Mol, canonical_smiles: str) -> Dict[int, Tuple[int, int]]:
    atom_spans = scan_smiles_atom_spans(canonical_smiles)
    output_order_text = mol.GetProp("_smilesAtomOutputOrder")
    output_order = list(ast.literal_eval(output_order_text))
    if len(atom_spans) != len(output_order):
        raise ValueError(
            "Could not align canonical SMILES atoms with RDKit output order. "
            f"Found {len(atom_spans)} spans for {len(output_order)} atoms."
        )
    return {atom_index: atom_spans[row_index] for row_index, atom_index in enumerate(output_order)}


def wordpiece_tokenize_with_offsets(tokenizer, text: str) -> List[Tuple[str, int, int]]:
    vocab = tokenizer.vocab
    unk_token = tokenizer.unk_token
    max_chars = tokenizer.wordpiece_tokenizer.max_input_chars_per_word
    pieces = []

    for match in re.finditer(r"\S+", text):
        token = match.group(0)
        offset = match.start()
        if len(token) > max_chars:
            pieces.append((unk_token, offset, offset + len(token)))
            continue

        start = 0
        sub_tokens = []
        is_bad = False
        while start < len(token):
            end = len(token)
            current = None
            current_span = None
            while start < end:
                substring = token[start:end]
                if start > 0:
                    substring = "##" + substring
                if substring in vocab:
                    current = substring
                    current_span = (offset + start, offset + end)
                    break
                end -= 1

            if current is None:
                is_bad = True
                break

            sub_tokens.append((current, current_span[0], current_span[1]))
            start = end

        if is_bad:
            pieces.append((unk_token, offset, offset + len(token)))
        else:
            pieces.extend(sub_tokens)

    return pieces


def token_positions_for_atoms(tokenizer, canonical_smiles: str, atom_spans: Dict[int, Tuple[int, int]]) -> Dict[int, List[int]]:
    prefixed = "[CLS]" + canonical_smiles
    offset = len("[CLS]")
    pieces = wordpiece_tokenize_with_offsets(tokenizer, prefixed)
    positions = {}

    for atom_index, (start, end) in atom_spans.items():
        token_positions = []
        span_start = start + offset
        span_end = end + offset
        for token_index, (_, token_start, token_end) in enumerate(pieces):
            if token_end > span_start and token_start < span_end:
                token_positions.append(token_index)

        if not token_positions:
            raise ValueError(f"Could not map atom {atom_index} to tokenizer positions.")
        positions[atom_index] = token_positions

    return positions


def get_atom_metadata(mol: Chem.Mol, atom_index: int) -> dict:
    atom = mol.GetAtomWithIdx(atom_index)
    return {
        "atom_index": atom_index,
        "atom_symbol": atom.GetSymbol(),
        "atomic_num": atom.GetAtomicNum(),
        "is_aromatic": bool(atom.GetIsAromatic()),
        "formal_charge": atom.GetFormalCharge(),
        "degree": atom.GetDegree(),
    }


def score_to_color(score: float, max_abs_score: float) -> tuple:
    if max_abs_score <= 0:
        return (0.85, 0.85, 0.85)

    intensity = min(abs(score) / max_abs_score, 1.0)
    fade = 0.75 * intensity
    if score >= 0:
        return (1.0, 1.0 - fade, 1.0 - fade)

    return (1.0 - fade, 1.0 - fade, 1.0)


def draw_atom_attribution(smiles: str,
                          attribution: pd.DataFrame,
                          output_path: str,
                          score_column: str = "contribution",
                          legend: str = "") -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid fluorophore SMILES: {smiles}")

    scores = attribution.set_index("atom_index")[score_column].to_dict()
    max_abs_score = max((abs(float(v)) for v in scores.values()), default=0.0)

    draw_mol = Chem.Mol(mol)
    highlight_atoms = []
    highlight_colors = {}
    highlight_radii = {}

    for atom in draw_mol.GetAtoms():
        atom_index = atom.GetIdx()
        score = float(scores.get(atom_index, 0.0))
        atom.SetProp("atomNote", f"{atom_index}:{score:.2f}")
        highlight_atoms.append(atom_index)
        highlight_colors[atom_index] = score_to_color(score, max_abs_score)
        highlight_radii[atom_index] = 0.32 + 0.22 * (abs(score) / max_abs_score if max_abs_score else 0.0)

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.legendFontSize = 24
    drawer.DrawMolecule(
        draw_mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(drawer.GetDrawingText())

    return output_path


def load_fluodb_model(checkpoint_path: str,
                      tokenizer,
                      device: torch.device,
                      config: Optional[dict] = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict containing 'state_dict': {checkpoint_path}")

    model_config = checkpoint.get("config", config)
    if model_config is None:
        model_config = {"bert_config_text": "./config_bert.json"}

    model = SPMM_fluodb_regressor(config=model_config, tokenizer=tokenizer)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain 'state_dict': {checkpoint_path}")

    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    value_mean = torch.as_tensor(checkpoint.get("value_mean", 0.0), dtype=torch.float).item()
    value_std = torch.as_tensor(checkpoint.get("value_std", 1.0), dtype=torch.float).item()
    return model, model_config, value_mean, value_std


def encode_text(tokenizer, text: str, device: torch.device, max_length: int):
    encoded = tokenizer(
        [text],
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return encoded.input_ids[:, 1:], encoded.attention_mask[:, 1:]


@torch.no_grad()
def predict_masked_batch(model,
                         tokenizer,
                         canonical_smiles: str,
                         canonical_solvent: str,
                         atom_positions: Dict[int, List[int]],
                         value_mean: float,
                         value_std: float,
                         device: torch.device,
                         max_length_smiles: int,
                         max_length_solvent: int) -> torch.Tensor:
    atom_indices = sorted(atom_positions)
    smiles_ids, smiles_mask = encode_text(tokenizer, "[CLS]" + canonical_smiles, device, max_length_smiles)
    solvent_ids, solvent_mask = encode_text(tokenizer, "[CLS]" + canonical_solvent, device, max_length_solvent)

    row_count = len(atom_indices) + 1
    smiles_ids = smiles_ids.repeat(row_count, 1)
    smiles_mask = smiles_mask.repeat(row_count, 1)
    solvent_ids = solvent_ids.repeat(row_count, 1)
    solvent_mask = solvent_mask.repeat(row_count, 1)

    for row_index, atom_index in enumerate(atom_indices, start=1):
        for token_position in atom_positions[atom_index]:
            if token_position >= smiles_ids.size(1) or smiles_mask[0, token_position].item() == 0:
                raise ValueError(
                    f"Atom {atom_index} was truncated by max_length_smiles={max_length_smiles}. "
                    "Increase --max_length_smiles."
                )
            smiles_ids[row_index, token_position] = tokenizer.unk_token_id

    preds = model(
        smiles_ids,
        smiles_mask,
        solvent_ids,
        solvent_mask,
        eval=True,
    )
    return preds.cpu() * value_std + value_mean


def explain_one_target(checkpoint_path: str,
                       target: str,
                       fluorophore_smiles: str,
                       solvent_smiles: str,
                       tokenizer,
                       device: torch.device,
                       atom_indices: Optional[Iterable[int]] = None,
                       output_csv: str = "",
                       output_png: str = "",
                       max_length_smiles: int = 160,
                       max_length_solvent: int = 64,
                       config: Optional[dict] = None,
                       include_generic_aliases: bool = True) -> pd.DataFrame:
    model, config, value_mean, value_std = load_fluodb_model(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )

    mol, canonical_smiles = canonicalize_smiles(fluorophore_smiles)
    _, canonical_solvent = canonicalize_smiles(solvent_smiles)
    atom_spans = canonical_atom_spans(mol, canonical_smiles)
    atom_positions = token_positions_for_atoms(tokenizer, canonical_smiles, atom_spans)

    all_atom_indices = sorted(atom_positions)
    if atom_indices is not None:
        selected = sorted(set(int(index) for index in atom_indices))
        invalid = sorted(set(selected) - set(all_atom_indices))
        if invalid:
            raise ValueError(f"Atom indices out of range: {invalid}")
        atom_positions = {index: atom_positions[index] for index in selected}

    predictions = predict_masked_batch(
        model=model,
        tokenizer=tokenizer,
        canonical_smiles=canonical_smiles,
        canonical_solvent=canonical_solvent,
        atom_positions=atom_positions,
        value_mean=value_mean,
        value_std=value_std,
        device=device,
        max_length_smiles=max_length_smiles,
        max_length_solvent=max_length_solvent,
    )

    baseline = float(predictions[0])
    records = []
    for row_index, atom_index in enumerate(sorted(atom_positions), start=1):
        masked_pred = float(predictions[row_index])
        contribution = baseline - masked_pred
        row = get_atom_metadata(mol, atom_index)
        row["fluorophore_smiles"] = canonical_smiles
        row["solvent_smiles"] = canonical_solvent
        row[f"baseline_{target}"] = baseline
        row[f"masked_{target}"] = masked_pred
        row[f"contribution_{target}"] = contribution
        row[f"abs_contribution_{target}"] = abs(contribution)

        if include_generic_aliases:
            row["baseline_prediction"] = baseline
            row["masked_prediction"] = masked_pred
            row["contribution"] = contribution
            row["abs_contribution"] = abs(contribution)

        records.append(row)

    result = pd.DataFrame(records)
    if len(result) > 0:
        rank_column = "abs_contribution" if include_generic_aliases else f"abs_contribution_{target}"
        result[f"importance_rank_{target}"] = result[rank_column].rank(method="dense", ascending=False).astype(int)
        if include_generic_aliases:
            result["importance_rank"] = result[f"importance_rank_{target}"]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result.to_csv(output_csv, index=False)

    if output_png:
        score_column = "contribution" if include_generic_aliases else f"contribution_{target}"
        draw_atom_attribution(
            smiles=canonical_smiles,
            attribution=result,
            output_path=output_png,
            score_column=score_column,
            legend=f"{target} baseline: {baseline:.3f}",
        )

    return result


def explain_all_targets(model_dir: str,
                        targets: Sequence[str],
                        fluorophore_smiles: str,
                        solvent_smiles: str,
                        tokenizer,
                        device: torch.device,
                        atom_indices: Optional[Iterable[int]] = None,
                        output_csv: str = "",
                        output_png_dir: str = "",
                        max_length_smiles: int = 160,
                        max_length_solvent: int = 64) -> pd.DataFrame:
    merged = None
    metadata_columns = [
        "fluorophore_smiles",
        "solvent_smiles",
        "atom_index",
        "atom_symbol",
        "atomic_num",
        "is_aromatic",
        "formal_charge",
        "degree",
    ]

    model_config = None
    for target in targets:
        checkpoint_path = os.path.join(model_dir, f"{target}_best.pth")
        output_png = ""
        if output_png_dir:
            os.makedirs(output_png_dir, exist_ok=True)
            output_png = os.path.join(output_png_dir, f"{safe_file_stem(fluorophore_smiles)}_{target}.png")

        target_df = explain_one_target(
            checkpoint_path=checkpoint_path,
            target=target,
            fluorophore_smiles=fluorophore_smiles,
            solvent_smiles=solvent_smiles,
            tokenizer=tokenizer,
            device=device,
            atom_indices=atom_indices,
            output_png=output_png,
            max_length_smiles=max_length_smiles,
            max_length_solvent=max_length_solvent,
            config=model_config,
            include_generic_aliases=False,
        )

        keep_columns = metadata_columns + [
            f"baseline_{target}",
            f"masked_{target}",
            f"contribution_{target}",
            f"abs_contribution_{target}",
            f"importance_rank_{target}",
        ]
        target_df = target_df.loc[:, keep_columns]

        if merged is None:
            merged = target_df
        else:
            merged = merged.merge(target_df, on=metadata_columns, how="inner")

    if merged is None:
        raise ValueError("No targets were provided.")

    overall_abs_columns = [f"abs_contribution_{target}" for target in targets]
    merged["max_abs_contribution"] = merged[overall_abs_columns].max(axis=1)
    merged["overall_importance_rank"] = merged["max_abs_contribution"].rank(method="dense", ascending=False).astype(int)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        merged.to_csv(output_csv, index=False)

    return merged


def parse_atom_indices(text: str) -> Optional[List[int]]:
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run atom-level token masking explainability for SPMM FluoDB regressors.")
    parser.add_argument("--checkpoint", help="Single-target checkpoint, e.g. output/FluoDB/emi_best.pth.")
    parser.add_argument("--model_dir", default="./output/FluoDB", help="Directory containing <target>_best.pth checkpoints.")
    parser.add_argument("--targets", nargs="+", default=["emi"], choices=["abs", "emi", "plqy", "e"], help="Targets to explain.")
    parser.add_argument("--fluorophore", help="Single fluorophore SMILES.")
    parser.add_argument("--fluorophores", nargs="+", help="One or more fluorophore SMILES.")
    parser.add_argument("--solvent", default="O", help="Solvent SMILES. Defaults to water.")
    parser.add_argument("--solvents", nargs="+", help="One solvent SMILES per fluorophore.")
    parser.add_argument("--use_examples", action="store_true", help="Run the built-in examples.")
    parser.add_argument("--atom_indices", default="", help="Optional comma-separated atom indices to explain.")
    parser.add_argument("--output_dir", default="./pred/spmm_fluodb_explain", help="Directory for CSV and PNG outputs.")
    parser.add_argument("--prefix", default="spmm_atom_explain", help="Output file prefix.")
    parser.add_argument("--no_images", action="store_true", help="Skip PNG generation.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific torch device.")
    parser.add_argument("--vocab_filename", default="./vocab_bpe_300.txt")
    parser.add_argument("--max_length_smiles", default=160, type=int)
    parser.add_argument("--max_length_solvent", default=64, type=int)
    args = parser.parse_args()

    fluorophores = []
    if args.fluorophore:
        fluorophores.append(args.fluorophore)
    if args.fluorophores:
        fluorophores.extend(args.fluorophores)

    entries = []
    if fluorophores:
        if args.solvents is not None:
            if len(args.solvents) != len(fluorophores):
                raise ValueError("The number of solvents must match the number of fluorophores.")
            solvents = list(args.solvents)
        else:
            solvents = [args.solvent] * len(fluorophores)
        entries.extend(zip(fluorophores, solvents))

    if args.use_examples or not entries:
        if len(EXAMPLE_FLUOROPHORES) != len(EXAMPLE_SOLVENTS):
            raise ValueError("EXAMPLE_FLUOROPHORES and EXAMPLE_SOLVENTS must have the same length.")
        entries.extend(zip(EXAMPLE_FLUOROPHORES, EXAMPLE_SOLVENTS))

    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    tokenizer = build_tokenizer(args.vocab_filename)
    atom_indices = parse_atom_indices(args.atom_indices)

    if len(args.targets) == 1:
        target = args.targets[0]
        checkpoint_path = args.checkpoint or os.path.join(args.model_dir, f"{target}_best.pth")
        tables = []
        image_paths = []
        for idx, (fluorophore, solvent) in enumerate(entries, start=1):
            entry_prefix = f"{args.prefix}_{idx:02d}"
            csv_path = os.path.join(args.output_dir, f"{entry_prefix}.csv")
            png_path = os.path.join(args.output_dir, f"{entry_prefix}_{safe_file_stem(solvent)}.png")
            result = explain_one_target(
                checkpoint_path=checkpoint_path,
                target=target,
                fluorophore_smiles=fluorophore,
                solvent_smiles=solvent,
                tokenizer=tokenizer,
                device=device,
                atom_indices=atom_indices,
                output_csv=csv_path,
                output_png="" if args.no_images else png_path,
                max_length_smiles=args.max_length_smiles,
                max_length_solvent=args.max_length_solvent,
            )
            result = result.copy()
            result.insert(0, "entry_id", idx)
            tables.append(result)
            if not args.no_images:
                image_paths.append(png_path)

        merged = pd.concat(tables, ignore_index=True)
        merged_csv_path = os.path.join(args.output_dir, f"{args.prefix}.csv")
        merged.to_csv(merged_csv_path, index=False)
        print(merged.sort_values(["entry_id", "abs_contribution"], ascending=[True, False]).head(20).to_string(index=False))
        print(f"\nSaved merged CSV to {merged_csv_path}")
        if not args.no_images:
            print("Saved figures:")
            for image_path in image_paths:
                print(image_path)
        return

    tables = []
    image_dirs = []
    for idx, (fluorophore, solvent) in enumerate(entries, start=1):
        example_dir = os.path.join(
            args.output_dir,
            f"{idx:02d}_{safe_file_stem(fluorophore)}__{safe_file_stem(solvent)}",
        )
        csv_path = os.path.join(args.output_dir, f"{args.prefix}_{idx:02d}.csv")
        result = explain_all_targets(
            model_dir=args.model_dir,
            targets=args.targets,
            fluorophore_smiles=fluorophore,
            solvent_smiles=solvent,
            tokenizer=tokenizer,
            device=device,
            atom_indices=atom_indices,
            output_csv=csv_path,
            output_png_dir="" if args.no_images else example_dir,
            max_length_smiles=args.max_length_smiles,
            max_length_solvent=args.max_length_solvent,
        )
        result = result.copy()
        result.insert(0, "entry_id", idx)
        tables.append(result)
        if not args.no_images:
            image_dirs.append(example_dir)

    merged = pd.concat(tables, ignore_index=True)
    csv_path = os.path.join(args.output_dir, f"{args.prefix}.csv")
    merged.to_csv(csv_path, index=False)

    rank_columns = [f"importance_rank_{target}" for target in args.targets]
    preview_columns = ["entry_id", "fluorophore_smiles", "solvent_smiles", "atom_index", "atom_symbol", "overall_importance_rank"] + rank_columns
    print(merged.loc[:, preview_columns].sort_values(["entry_id", "overall_importance_rank"]).head(30).to_string(index=False))
    print(f"\nSaved CSV to {csv_path}")
    if not args.no_images:
        print("Saved example images under:")
        for image_dir in image_dirs:
            print(image_dir)


if __name__ == "__main__":
    main()
