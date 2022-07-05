# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
from operator import itemgetter
from sklearn import cluster
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
from PIL import Image
import requests
import json
import time
from st_aggrid import AgGrid
from collections import Counter, OrderedDict
from ast import literal_eval
import numpy as np
from streamlit_echarts import st_echarts
from streamlit_echarts import JsCode
from traitlets import All


#---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(page_title="GISP результаты кластеризации",
                   page_icon="📈", layout="wide")
#---------------------------------#

cover_image = Image.open(requests.get(
    "https://ecpexpert.ru/wp-content/uploads/2021/05/lichnyj-kabinet-gisp-minpromtorg-funktsional-akkaunta-pravila-vosstanovleniya-parolya.jpg", stream=True).raw)
st.image(cover_image)


st.title('Результаты кластеризации')
st.markdown("""
Приложение создано исключительно с целью предоставить результаты экспериментов!
""")


# Table with metrics Siluette, DBCV
# header
st.markdown('Полученные метрики кластеризации для 100тыс. товаров')

results = pd.read_csv("results.csv")
st.table(results)


# Полученные шаблоны
product_names = pd.read_csv('products_name.csv')


@st.cache(allow_output_mutation=True)
def get_products_df(path):

    products_df = pd.read_csv(path)
    products_df['okpd2'].fillna(0, inplace=True)
    products_df['ktru'].fillna(0, inplace=True)

    return products_df


@st.cache(allow_output_mutation=True)
def get_okpd2_ktru():
    okpd2_df = pd.read_csv("okpd2.csv", index_col="id")
    okpd2_df.loc[0] = {'name': "Не определено"}
    ktru_df = pd.read_csv("ktru.csv", index_col="id")
    ktru_df.loc[0] = {'name': "Не определено"}
    return okpd2_df, ktru_df


@st.cache(allow_output_mutation=True)
def get_data(products_df):
    return products_df['data'].values.tolist()


okpd2_df, ktru_df = get_okpd2_ktru()
df = get_products_df('products_df_rubert-tiny.csv')

df = df.merge(okpd2_df, left_on='okpd2',
              right_on='id', suffixes=('', '_okpd2'))

df = df.merge(ktru_df, left_on='ktru', right_on='id', suffixes=('', '_ktru'))
df = df.merge(product_names, on='id', suffixes=('', '_product'))

df['okpd2'].fillna(0, inplace=True)
df['ktru'].fillna(0, inplace=True)


cluster_groups = np.sort(df.group_number.unique())


def make_template(df: pd.DataFrame, cluster_num: int):

    group = df[df.group_number == cluster_num]
    unique_clusters = group.cluster_number.unique()
    result = []
    for cluster_num in unique_clusters:
        template = dict()
        for field in group[group.cluster_number == cluster_num]['fields']:

            cur_field = literal_eval(field)

            for key in cur_field.keys():

                if key in template:
                    template[key] += 1
                else:
                    template[key] = 1
        result.append({cluster_num: template})
    return result


#result_df = df[['fields', 'name', 'name_ktru']]
#result_df.columns = ['Характеристики', 'OKPD_2', 'KTRU']
# AgGrid(result_df)


st.subheader('Строим Шаблон')

okpd2_ids = df.okpd2.unique()

okpd2 = st.selectbox("ОКПД2",
                     options=okpd2_ids,
                     format_func=lambda x: okpd2_df.loc[int(
                         x)][['code', 'name']].values,
                     )


okpd2_products_table = df[df.okpd2 == int(okpd2)]

ktru_ids = okpd2_products_table.ktru.unique()
ktru = st.selectbox("КТРУ",
                    options=ktru_ids,
                    format_func=lambda x: ktru_df.loc[int(
                        x)][['code', 'name']].values,
                    )
ktru_products_table = okpd2_products_table[okpd2_products_table.ktru == ktru]


group_number = ktru_products_table['group_number'].values[0]

template_result = make_template(df, group_number)

cluster_keys={}
for ind, item in enumerate(template_result):
    cluster_keys[ind] = list(item.keys())[0]

inv_cluster_keys = {v: k for k, v in cluster_keys.items()}



cluster_num = st.selectbox(f'Кластер № (Всего кластеров {len(cluster_keys)})',
                           options=list(cluster_keys.keys()),
                           )              

real_cluter_num = cluster_keys[cluster_num]

cur_template = sorted(list(template_result[cluster_num].values())[
                      0].items(), key=itemgetter(1), reverse=True)

#cur_template = sorted(list(template_result[cluster_num].values())[0].items(), key=lambda x: x[1], reverse=True)

st.subheader(f'Всего характеристик в данном шаблоне: {len(cur_template)}')

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

character = pd.DataFrame(cur_template)
character.columns = ['Тип характеристики', 'Частота встречаемости']
st.table(character)


st.subheader(f'Товары из этого кластера: Всего { ktru_products_table[ktru_products_table["cluster_number"] == real_cluter_num].shape[0]}')
st.dataframe(
    ktru_products_table[ktru_products_table['cluster_number'] == real_cluter_num][['id', 'name_product']])




st.subheader('Итого:')
st.table(pd.DataFrame(ktru_products_table['cluster_number'].value_counts().rename_axis('Кластер').reset_index(name='Количество товаров')).replace({'Кластер': inv_cluster_keys}))
