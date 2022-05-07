from pathlib import Path

# Folder structure is
# ROOT_PATH/SIZE/SENSOR/[left,right]/Normalized[Images,Masks]/*.bmp
ROOT_PATH = Path().cwd().parent
NDIRIS_240x20 = ROOT_PATH / 'NDIRIS_240x20'
NDIRIS_240x40 = ROOT_PATH / 'NDIRIS_240x40'
LG4000 = 'LG4000'
LG2200 = 'LG2200'
SENSORS = (LG4000, LG2200)


def dataset_summary(out_folder='experiments/ndiris_summary/'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    images_list = list(ROOT_PATH.glob('*/*/*/NormalizedImages/*.bmp'))
    df = pd.DataFrame({'path': images_list})
    df['size'] = df['path'].apply(
        lambda x: list(x.parents)[3].name.split('_')[-1])
    df['sensor'] = df['path'].apply(
        lambda x: list(x.parents)[2].name)
    df['eye'] = df['path'].apply(
        lambda x: list(x.parents)[1].name)
    df['id_number'] = df['path'].apply(
        lambda x: int(x.stem.split('d')[0]))

    older_df = pd.read_csv(NDIRIS_240x20 / 'dataframe_old.csv')
    labels_csvs_list = list(ROOT_PATH.glob('*/*/*/labels.csv'))
    labels_dfs = [pd.read_csv(f) for f in labels_csvs_list]
    labels_df = pd.concat(labels_dfs)

    def find_gender_in_older_df(img_path):
        img_name = img_path.stem.split('_')[0]
        labels = labels_df[labels_df.img_id == img_name].gender.values
        if len(labels) != 0:
            gender = "Male" if labels[0] == 0 else "Female"
            return gender
        img_id = int(img_path.stem.split('d')[0])
        labels_older = older_df[older_df.id_number == img_id].gender.dropna()

        return labels_older.values[0]

    df['gender'] = df.path.apply(find_gender_in_older_df)

    # Statistics
    n_subjects = len(df.id_number.unique())
    print(f'Number of subjects: {n_subjects}')
    male_subjects = len(df[df.gender == "Male"].id_number.unique())
    print(f'\tMale subjects: {male_subjects}')
    female_subjects = len(df[df.gender == "Female"].id_number.unique())
    print(f'\tFemale subjects: {female_subjects}\n')

    images_per_subject = df.groupby('id_number').count().path.describe()
    print(f'Images per subject: \n{images_per_subject}\n')
    male_images = len(df[df.gender == 'Male'])
    print(f'Male images: \n{male_images}\n')
    female_images = len(df[df.gender == 'Female'])
    print(f'Female images: \n{female_images}\n')
    images_per_sensor = df.groupby('sensor').count().path
    print(f'Images per sensor: \n{images_per_sensor}\n')

    with sns.plotting_context('talk'):
        gender_hist = df.groupby('id_number').gender.describe()
        sns.histplot(gender_hist, x='count', hue='top', binwidth=10)
        plt.legend(title='gender', loc='upper right',
                   labels=['Male', 'Female'])
        plt.xlabel('Number of images')
        plt.title('Number of images per subject, NDIris Dataset')
        plt.tight_layout()
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_folder / 'images_histogram.png')

    return df, labels_df, older_df
