import numpy as np
import matplotlib.pyplot as plt
import psi4

# Memory and output setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('water_uvvis.dat', False)

# 1. Define geometry
# mol = psi4.geometry("""
# 0 1
# O   0.00000000   0.00000000   0.11726900
# H   0.00000000   0.75696800  -0.46907600
# H   0.00000000  -0.75696800  -0.46907600
# symmetry c1
# """)
mol = psi4.geometry("""
    pubchem:benzene
""")

# 2. Configure calculation options
n_states = 5
psi4.set_options({
    'basis': 'def2-svp',
    'reference': 'rks',
    'tdscf_states': [n_states],  # Request excited states
})

# 3. Perform TD-DFT calculation
energy, wfn = psi4.energy('td-b3lyp', return_wfn=True)

# 4. Extract excitation energies and oscillator strengths dynamically
vars_dict = wfn.variables()

ex_energies = []
osc_strengths = []

# Fetch excitation energies (in eV) and oscillator strengths from wfn variables
for i in range(1, n_states + 1):
    # Search keys matching root i
    e_key = next((k for k in vars_dict.keys() if f'ROOT {i}' in k and 'EXCITATION ENERGY' in k), None)
    f_key = next((k for k in vars_dict.keys() if f'ROOT {i}' in k and 'OSCILLATOR STRENGTH' in k), None)
    
    if e_key and f_key:
        # Psi4 excitation energy variable values are in Hartrees or eV depending on key;
        # Check if key explicitly ends with (EV)
        e_val = vars_dict[e_key]
        if '(EV)' not in e_key.upper() and 'EV' not in e_key.upper():
            e_val *= 27.211386  # convert Hartree to eV
            
        ex_energies.append(e_val)
        osc_strengths.append(vars_dict[f_key])

print("Excitation Energies (eV):", ex_energies)
print("Oscillator Strengths:", osc_strengths)

# 5. Spectrum Gaussian Broadening
wavelengths = np.linspace(50, 400, 1000)  # Water absorbs in VUV (<200 nm)
energies_grid = 1240.0 / wavelengths

sigma = 0.4  # Broadening parameter (eV)
spectrum = np.zeros_like(energies_grid)

for energy_ev, osc in zip(ex_energies, osc_strengths):
    spectrum += osc * np.exp(-((energies_grid - energy_ev) ** 2) / (2 * sigma ** 2))

# 6. Plotting
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, spectrum, color='navy', lw=2, label='TD-B3LYP / def2-SVP')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (arb. units)')
plt.title('Calculated UV-Vis Spectrum')
plt.gca()
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
