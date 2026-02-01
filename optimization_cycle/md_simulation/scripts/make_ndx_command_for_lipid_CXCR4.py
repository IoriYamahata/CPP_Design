#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gro",
        type=str,
        help="input gro file",
        required=True,
    )
    return parser


def gen_receptor_name(gro_file_path: str) -> str:
    """
    gro の親ディレクトリ名から receptor_name を作る。
    例: /path/to/CXCR4_RWWSWWWWWWWRWWK-1/run/start.gro
        -> 親dir名 "CXCR4_RWWSWWWWWWWRWWK-1"（run の上を使うなら適宜調整）
    現状は gro が入っているディレクトリ名を使う。
    """
    directory_path = os.path.dirname(gro_file_path)
    folder_name = os.path.basename(directory_path)
    receptor_name = folder_name
    return receptor_name


def get_last_residue_number(gro_file_path: str) -> int:
    """
    .gro の最後の原子行（末尾のボックス行の1つ前）から残基番号を取る。
    """
    with open(gro_file_path, "r") as gro_file:
        lines = gro_file.readlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid .gro file (too short): {gro_file_path}")
        last_atom_line = lines[-2]
        residue_number = int(last_atom_line[0:5].strip())
    return residue_number


def get_residue_length_from_name(receptor_name: str) -> int:
    """
    receptor_name 例: "CXCR4_RWWSWWWWWWWRWWK-1"
      -> len("RWWSWWWWWWWRWWK") を返す

    期待フォーマット:
      <RECEPTOR>_<PEPTIDESEQ>-<ID>
    - 末尾の "-数字" は無視
    - "_" より後ろを配列として扱う
    """
    base = re.sub(r"-\d+$", "", receptor_name)

    if "_" not in base:
        raise ValueError(f"Invalid name (missing '_'): {receptor_name}")

    seq = base.split("_", 1)[1].strip()
    if not seq:
        raise ValueError(f"Empty sequence part: {receptor_name}")

    # 1文字アミノ酸表記（必要なら拡張）
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
        raise ValueError(f"Sequence part contains invalid characters: {seq}")

    return len(seq)


def make_ndx_command_string(last_residue_length: int) -> str:
    """
    要望どおりハードコード：
      Receptor: r 26-304
      Ligand  : r 0-(last_residue_length+1)
    """
    ndx  = "r 26-304\n"
    ndx += "name 10 Receptor\n"
    ndx += f"r 0-{last_residue_length + 1}\n"
    ndx += "name 11 Ligand\n"
    ndx += "q\n"
    return ndx


def main():
    parser = get_parser()
    args = parser.parse_args()

    receptor_name = gen_receptor_name(args.gro)

    # gro から取る値（今は使わないが、検証やログに残せる）
    last_residue_number = get_last_residue_number(args.gro)

    # ファイル名（フォルダ名）から配列長を取る
    last_residue_length = get_residue_length_from_name(receptor_name)

    ndx_command_string = make_ndx_command_string(last_residue_length)

    with open("commands.in", "w") as f:
        f.write(ndx_command_string)

    print(f"[info] receptor_name       : {receptor_name}")
    print(f"[info] last_residue_number : {last_residue_number}")
    print(f"[info] ligand_seq_length   : {last_residue_length}")
    print("[info] wrote commands.in")


if __name__ == "__main__":
    sys.exit(main())
