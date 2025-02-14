# Virtual Database Construction and Machine-Learning-Assisted High-Throughput Evaluation of Amorphous Porous Carbon Materials as Iodine Sorbents
## Introduction
QAPC is a materials structure database designed to facilitate the exploration of amorphous porous carbon (APC) materials for iodine adsorption applications. It contains a total of 19,599 distinct models of amorphous porous carbon, each characterized by unique structural and chemical properties. These models were generated using advanced simulation techniques to capture a wide range of possible configurations and pore structures, making the database a helpful resource for researchers investigating the potential of APC materials in environmental and industrial applications. The database is not only a compilation of material structures but also a critical tool for machine learning-assisted high-throughput screening, enabling rapid evaluation of the iodine adsorption capacity and efficiency of different APC models. This high-throughput approach, combined with advanced computational methods, allows for the identification of the most promising APC candidates, accelerating the development of materials with superior iodine adsorption capabilities.
## Liquid Quenching Process for APC Model Generation
This process utilizes the Hotpot software package, developed by our research group, to simulate the "liquid quenching" method for generating APC models. The process is executed within Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS), so please ensure that LAMMPS is available on your system. In this example, we demonstrate how to generate a pure carbon APC model with a density of 0.85 g/cm³ using the Tersoff force field.
### Installation
Make sure to install the Hotpot package and LAMMPS before running the process.
### Example Usage
```
from hotpot.cheminfo import Molecule
# Set temperature values
t0 = 298.15  # T0 (K)
t1 = 9500    # T1 (K)
t2 = 3500    # T2 (K)
# Define file paths for saving results
dump_path = './output/mq_0.85_1_3500_9500.xyz'
save_path = './output/mq_0.85_1_3500_9500.cif'
# Start the liquid quenching process to generate the APC model
mol = Molecule.create_aCryst_by_mq(
    elements={'C': 1.0},   # Element composition ratio (pure carbon in this case)
    force_field='aMaterials/SiC.tersoff',  # Relative path to the force field (ensure proper force field for accuracy)
    density=0.85,  # Target density for the generated APC
    a=50, b=50, c=50,   # Lattice constants for the crystal
    origin_temp=t0,
    highest_temp=t1,
    melt_temp=t2,
    ff_args=('C',),     # Force field-specific arguments (refer to LAMMPS pair_coeff)
    path_dump_to=dump_path   # Path to save the trajectory of the liquid-quench process
)
# Save the generated molecule as a crystal structure file
mol.crystal().space_group = 'P1'
mol.writefile('cif', save_path)
```
### Key Steps
      Elements: Define the element composition for your APC model (in this case, pure carbon).
      Force Field: Specify the force field used for the simulation (ensure compatibility with LAMMPS).
      Density: Set the target density of the APC model.
      Temperature: Control the temperature parameters for the liquid quenching process (T0, T1, T2).
      File Paths: Specify where to save the trajectory and the final crystal structure.
### Output
      Trajectory File (.xyz format): Contains the atomic positions throughout the quenching process.
      Crystal Structure File (.cif format): Contains the final structure of the APC model, ready for further analysis.
By following this procedure, you can generate APC models for various applications, including adsorption studies.
