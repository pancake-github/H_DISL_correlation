#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import (
    SelectTypeModifier,
    DeleteSelectedModifier,
    DislocationAnalysisModifier,
)


def compute_dislocation_density(dump_path: str, max_frames: int | None = None):
    """Compute engineering strain and dislocation density for each frame.

    Parameters
    ----------
    dump_path : str
        Path to the LAMMPS dump file.
    max_frames : int, optional
        Process at most this many frames for quick testing.

    Returns
    -------
    strain : np.ndarray
        Engineering strain for each processed frame.
    density : np.ndarray
        Dislocation density (m^-2) for each processed frame.
    """
    pipeline = import_file(dump_path, multiple_frames=True)

    # Remove hydrogen atoms (type 2)
    pipeline.modifiers.append(SelectTypeModifier(types={2}))
    pipeline.modifiers.append(DeleteSelectedModifier())

    # Dislocation analysis for BCC iron
    dxa = DislocationAnalysisModifier()
    dxa.input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC
    pipeline.modifiers.append(dxa)

    nframes = pipeline.source.num_frames
    if max_frames is not None:
        nframes = min(nframes, max_frames)

    strains = []
    densities = []

    Lz_active0 = None

    for i in range(nframes):
        data = pipeline.compute(i)
        cell = data.cell
        lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]

        Lz_active = lz - 15.0 - 38.5
        if Lz_active0 is None:
            Lz_active0 = Lz_active
        strain = (Lz_active - Lz_active0) / Lz_active0

        A = lx * ly
        V = A * Lz_active

        line_length = data.attributes.get(
            "DislocationAnalysis.total_line_length", 0.0
        )
        rho = (line_length / V) * 1e20  # Convert from Å^-2 to m^-2

        strains.append(strain)
        densities.append(rho)

    return np.asarray(strains), np.asarray(densities)


def main():
    parser = argparse.ArgumentParser(
        description="Compute dislocation density as a function of engineering strain."
    )
    parser.add_argument("--dump", required=True, help="Path to LAMMPS dump file")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit number of frames"
    )
    args = parser.parse_args()

    strain, density = compute_dislocation_density(args.dump, args.max_frames)

    plt.figure()
    plt.plot(strain, density, marker="o")
    plt.xlabel("Engineering strain")
    plt.ylabel("Dislocation density (m$^{-2}$)")
    plt.tight_layout()
    plt.savefig("dislocation_density_vs_strain.png")

    # Show first few data points
    for s, d in list(zip(strain, density))[:5]:
        print(f"strain={s:.6f}, density={d:.3e}")


if __name__ == "__main__":
    main()
