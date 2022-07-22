'''Contains some class variables so as to not clutter magentro.py.'''

import numpy as np

# Order is important! Treat all as ordered dicts.
# The dict values are safe to change, but not the keys.

RAW_DF_COLUMNS = {
    'T': 'T',
    'H': 'H',
    'M': 'M',
    'M_err': 'M_err',
    'M_per_mass': 'M_per_mass',
    'M_per_mass_err': 'M_per_mass_err',
    'dM_dT': 'dM_dT',
    'Delta_SM': 'Delta_SM'
}

RAW_DF_DEFAULT_UNITS = {
    'T': 'K',
    'H': 'Oe',
    'M': 'emu',
    'M_err': 'emu',
    'M_per_mass': 'emu/g',
    'M_per_mass_err': 'emu/g',
    'dM_dT': 'cal/K/g/Oe',
    'Delta_SM': 'cal/K/g'
}

# INITIAL_UNITS contains dM_dT and Delta_SM units if they were to be obtained
# directly from the default units for T, H, and M (rather than the standard
# cgs energy units)
INITIAL_UNITS = {
    'T': 'K',
    'H': 'Oe',
    'M': 'emu',
    'M_err': 'emu',
    'M_per_mass': 'emu/g',
    'M_per_mass_err': 'emu/g',
    'dM_dT': 'emu/K/g',
    'Delta_SM': 'emu*Oe/K/g'
}

CONVERTED_DF_COLUMNS = {
    'T': 'T',
    'H': 'H',
    'M': 'M',
    'M_err': 'M_err',
    'M_per_mass': 'M_per_mass',
    'M_per_mass_err': 'M_per_mass_err',
    'dM_dT': 'dM_dT',
    'Delta_SM': 'Delta_SM'
}

CONVERTED_DF_UNITS = {
    'T': 'K',
    'H': 'T',
    'M': 'A*m^2',
    'M_err': 'A*m^2',
    'M_per_mass': 'A*m^2/kg',
    'M_per_mass_err': 'A*m^2/kg',
    'dM_dT': 'J/K/kg/T',
    'Delta_SM': 'J/K/kg'
}

PROCESSED_DF_COLUMNS = {
    'T': 'T',
    'H': 'H',
    'M': 'M',
    'M_err': 'M_err',
    'M_per_mass': 'M_per_mass',
    'M_per_mass_err': 'M_per_mass_err',
    'dM_dT': 'dM_dT',
    'Delta_SM': 'Delta_SM'
}

PROCESSED_DF_UNITS = {
    'T': CONVERTED_DF_UNITS['T'],
    'H': CONVERTED_DF_UNITS['H'],
    'M': CONVERTED_DF_UNITS['M'],
    'M_err': CONVERTED_DF_UNITS['M'],
    'M_per_mass': CONVERTED_DF_UNITS['M_per_mass'],
    'M_per_mass_err': CONVERTED_DF_UNITS['M_per_mass_err'],
    'dM_dT': CONVERTED_DF_UNITS['dM_dT'],
    'Delta_SM': CONVERTED_DF_UNITS['Delta_SM']
}

PROPORTIONAL = lambda x: x

INVERSE = lambda x: 1.0 / x

# read: <first key> changes in response to <second key> as described by <value>
# changes to the T columns are handled separately
CONVERSIONS = {
    'H': {
        'H': PROPORTIONAL
    },
    'M': {
        'M': PROPORTIONAL
    },
    'M_err': {
        'M': PROPORTIONAL
    },
    'M_per_mass': {
        'M': PROPORTIONAL,
        'sample_mass': INVERSE
    },
    'M_per_mass_err': {
        'M': PROPORTIONAL,
        'sample_mass': INVERSE
    },
    'dM_dT': {
        'T': INVERSE,
        'M': PROPORTIONAL,
        'sample_mass': INVERSE
    },
    'Delta_SM': {
        'T': INVERSE,
        'H': PROPORTIONAL,
        'M': PROPORTIONAL,
        'sample_mass': INVERSE
    }
}

DEFAULT_PRESETS = {
    'npoints': np.int64(1000),
    'temp_range': np.array([-np.inf, np.inf], dtype=np.float64, ndmin=1),
    'fields': np.array([], dtype=np.float64, ndmin=1),
    'decimals': np.int64(5),
    'max_diff': np.inf,
    'min_sweep_len': np.int64(10),
    'd_order': np.int64(2),
    'lmbds': np.array([np.nan], dtype=np.float64, ndmin=1),
    'lmbd_guess': np.float64(1e-4),
    'weight_err': True,
    'match_err': False,
    'min_kwargs': {
        'method': 'Nelder-Mead',
        'bounds': ((-np.inf, np.inf),),
        'options': {'maxfev': 50, 'xatol': 1e-2, 'fatol': 1e-6}
    },
    'add_zeros': False
}
