#!/usr/bin/env python
"""
"""
import argparse
from typing import Final
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import uproot.writing


BRANCH_LIST: Final[list[str]] = [
    # EFlowTrack
    "EFlowTrack/EFlowTrack.PT",
    "EFlowTrack/EFlowTrack.Eta",
    "EFlowTrack/EFlowTrack.Phi",
    "EFlowTrack/EFlowTrack.PID",
    "EFlowTrack/EFlowTrack.Charge",
    "EFlowTrack/EFlowTrack.IsRecoPU", # TODO
    # EFlowPhoton
    "EFlowPhoton/EFlowPhoton.ET",
    "EFlowPhoton/EFlowPhoton.Eta",
    "EFlowPhoton/EFlowPhoton.Phi",
    # EFlowNeutralHadron
    "EFlowNeutralHadron/EFlowNeutralHadron.ET",
    "EFlowNeutralHadron/EFlowNeutralHadron.Eta",
    "EFlowNeutralHadron/EFlowNeutralHadron.Phi",
    # MET
    "MissingET/MissingET.MET",
    "MissingET/MissingET.Phi",
    # PUPPI MET
    "PuppiMissingET/PuppiMissingET.MET",
    "PuppiMissingET/PuppiMissingET.Phi",
    # Generated MET
    "GenMissingET/GenMissingET.MET",
    "GenMissingET/GenMissingET.Phi",
    #
    "GenPileUpMissingET/GenPileUpMissingET.MET",
    "GenPileUpMissingET/GenPileUpMissingET.Phi",
]

PREFIX_ALIAS_DICT: Final[dict[str, str]] = {
    'EFlowTrack': 'track',
    'EFlowPhoton': 'photon',
    'EFlowNeutralHadron': 'neutral_hadron',
    'MissingET': 'pf_met',
    'PuppiMissingET': 'puppi_met',
    'GenMissingET': 'gen_met',
    'GenPileUpMissingET': 'gen_pileup_met',
}

FEATURE_ALIAS_DICT: Final[dict[str, str]] = {
    'ET': 'PT',
    'MET': 'PT',
    'IsRecoPU': 'is_reco_pu',
}


def make_alias(branch: str) -> str:
    prefix, feature = branch.split('/')[1].split('.')
    prefix = PREFIX_ALIAS_DICT[prefix]
    feature = FEATURE_ALIAS_DICT.get(feature, feature)
    feature = feature.lower()
    alias = f'{prefix}_{feature}'
    return alias


def run(input_path: Path,
        output_path: Path,
        input_treepath: str = 'Delphes',
        input_branch_list: list[str] = BRANCH_LIST,
        output_treepath: str = 'tree',
):
    print(f'read {input_path}:{input_treepath}')
    tree = uproot.open({input_path: input_treepath})

    aliases: dict[str, str] = {make_alias(each): each
                               for each in input_branch_list}
    expressions: list[str] = list(aliases.keys())
    data = tree.arrays(
        expressions=expressions,
        aliases=aliases,
    )
    print(f'got {len(data)} events')

    output = {}

    # track
    for feature in ['pt', 'eta', 'phi']:
        key = f'track_{feature}'
        output[key] = ak.values_astype(data[key], np.float32)

    for feature in ['pid', 'charge', 'is_reco_pu']:
        key = f'track_{feature}'
        output[key] = ak.values_astype(data[key], np.int64)

    track_abs_pid = np.abs(data['track_pid'])

    output['track_is_electron'] = ak.values_astype(track_abs_pid == 11, np.int64)
    output['track_is_muon'] = ak.values_astype(track_abs_pid == 13, np.int64)
    output['track_is_hadron'] = ak.values_astype(
        array=((track_abs_pid != 11) & (track_abs_pid != 13)),
        to=np.int64
    )

    # tower
    for feature in ['pt', 'eta', 'phi']:
        arrays = [ak.values_astype(data[f'{prefix}_{feature}'], np.float32)
                  for prefix in ['neutral_hadron', 'photon']]
        output[f'tower_{feature}'] = ak.concatenate(
            arrays=arrays,
            axis=1
        )

    data['neutral_hadron_is_hadron'] = ak.ones_like(data['neutral_hadron_pt'], dtype=np.int64)
    data['photon_is_hadron'] = ak.zeros_like(data['photon_pt'], dtype=np.int64)
    for feature in ['is_hadron']:
        arrays = [data[f'{prefix}_{feature}']
                  for prefix in ['neutral_hadron', 'photon']]
        output[f'tower_{feature}'] = ak.concatenate(
            arrays=arrays,
            axis=1
        )

    for obj in ['pf_met', 'puppi_met', 'gen_met', 'gen_pileup_met']:
        for feature in ['pt', 'phi']:
            key = f'{obj}_{feature}'
            output[key] = ak.values_astype(ak.flatten(data[key]), np.float32)

    print(f'write {output_path}')
    output_file = uproot.writing.recreate(output_path)
    output_file[output_treepath] = output
    output_file.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', '--input-path', required=True, type=Path, help='Help text')
    parser.add_argument('-o', '--output-path', required=True, type=Path, help='Help text')
    args = parser.parse_args()

    run(
        input_path=args.input_path,
        output_path=args.output_path
    )

if __name__ == '__main__':
    main()
