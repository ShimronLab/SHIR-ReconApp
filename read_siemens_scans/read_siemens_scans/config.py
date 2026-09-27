############### CONFIG ###############

DIM_DESCRIPTION = {
    'Par': 'Axis 1 - probably phase encoding 1',
    'Lin': 'Axis 2 - possibly readout',
    'Cha': 'Coil',
    'Col': 'Axis 3 - probably phase encoding 2'
}


############### END OF CONFIG ###############

for key, val in DIM_DESCRIPTION.items():
    DIM_DESCRIPTION.update({key: f'{val} ({key})'})
