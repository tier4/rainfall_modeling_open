import os


def exp_dir_exists(name:str):
    if os.path.isdir(name + '_experiment'):
        return True
    else:
        return False


def create_dir_structure(name:str):

    dir_root = name + '_experiment'

    dirs = []
    dirs.append(os.path.join(dir_root, 'trained_models'))
    dirs.append(os.path.join(dir_root, 'regression_problems'))
    dirs.append(os.path.join(dir_root, 'raw_data', 'crop_box_10'))
    dirs.append(os.path.join(dir_root, 'raw_data', 'crop_box_15'))
    dirs.append(os.path.join(dir_root, 'raw_data', 'crop_box_20'))
    dirs.append(os.path.join(dir_root, 'datasets'))

    for dir in dirs:
        os.makedirs(dir)


if __name__ == '__main__':

    # Check that directories do not already exist
    if exp_dir_exists('primary'):
        print('ERROR: \'primary\' experiment directory already exists!')
        exit()
    if exp_dir_exists('secondary'):
        print('ERROR: \'secondary\' experiment directory already exists!')
        exit()

    create_dir_structure('primary')
    create_dir_structure('secondary')
