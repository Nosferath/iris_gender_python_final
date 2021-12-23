from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


@st.cache
def get_csv(path):
    return pd.read_csv(path, index_col=0)


def main():
    st.title('Visualización de selección de parámetros XGBoost')
    dfs_list = list(Path('streamlit_dfs').glob('*.csv'))
    dfs_select = st.selectbox(
        'Elegir datos a mostrar',
        dfs_list,
        format_func=lambda x: x.stem
    )
    df = get_csv(dfs_select)
    params = [c for c in df.columns if c.startswith('param_')]
    if 'param_learning_rate' in params:
        hue_col = 'param_learning_rate'
        x_col = 'param_n_estimators'
        params = ['param_learning_rate', 'n_estimators_rank']
        df.loc[:, 'param_learning_rate'] = \
            df.loc[:, 'param_learning_rate'].round(3)
    else:
        hue_col = params[0]
        x_col = params[1]
    if 'param_min_child_weight' in params:
        df.loc[:, 'param_min_child_weight'] = \
            df.loc[:, 'param_min_child_weight'].round(1)
    datasets = list(df.dataset.unique())
    dataset_select = st.selectbox(
        'Elegir datasets a mostrar',
        datasets
    )
    data = df[df.dataset == dataset_select]
    if 'param_learning_rate' in params:
        data['n_estimators_rank'] = data.groupby(
            'param_learning_rate')['param_n_estimators'].rank(
            'dense', ascending=True)
        x_col = 'n_estimators_rank'
    param_selects = {}
    for param in params:
        param_selects[param] = st.multiselect(
            f'Valores a mostrar para {param}',
            list(data[param].unique()),
            list(data[param].unique())
        )
        data = data[data[param].isin(param_selects[param])]
    swap_params = st.checkbox(
        'Alternar parámetros'
    )
    enable_legend = st.checkbox(
        'Activar leyenda',
        True
    )
    color_legend = st.checkbox(
        'Colorear por std',
        False
    )
    if swap_params:
        temp = x_col
        x_col = hue_col
        hue_col = temp
    if not (dataset_select and dfs_select and all(param_selects)):
        st.error('Faltan opciones por seleccionar')
    else:
        # Generate grid plot
        reindexed = data.groupby([x_col, hue_col]
                                 ).accuracy.describe().reset_index()
        reindexed['annot'] = reindexed[['mean', 'std']].apply(
            lambda x: f'{100*x[0]:.2f}\n± {100*x[1]:.2f}', axis=1
        )
        if not color_legend:
            pivoted = reindexed.pivot(index=x_col, columns=hue_col,
                                      values='mean') * 100
        else:
            pivoted = reindexed.pivot(index=x_col, columns=hue_col,
                                      values='std') * 100
        pivoted_annot = reindexed.pivot(index=x_col, columns=hue_col,
                                        values='annot')
        sizex1 = st.number_input('Tamaño horizontal de gráfico', value=12)
        sizey1 = st.number_input('Tamaño vertical de gráfico', value=12)
        with sns.axes_style('whitegrid'), sns.plotting_context('talk'):
            fig = plt.figure(figsize=(sizex1, sizey1))
            sns.heatmap(pivoted, cmap='jet', annot=pivoted_annot, fmt='')
            if not enable_legend:
                plt.legend([], [], frameon=False)
            plt.title(f'Parameters for dataset {dataset_select}')
            st.pyplot(fig)
        # Generate box plots
        sizex2 = st.number_input('Tamaño horizontal de gráfico', value=18)
        sizey2 = st.number_input('Tamaño vertical de gráfico', value=8)
        with sns.axes_style('whitegrid'), sns.plotting_context('talk'):
            fig = plt.figure(figsize=(sizex2, sizey2))
            sns.boxplot(data=data, x=x_col, y='accuracy', hue=hue_col,
                        notch=True)
            if not enable_legend:
                plt.legend([], [], frameon=False)
            plt.title(f'Parameters for dataset {dataset_select}')
            st.pyplot(fig)

        # Display dataframe
        grouped = data.groupby(params)
        acc_desc = grouped.accuracy.describe()
        st.dataframe(acc_desc.reset_index())



if __name__ == '__main__':
    main()
